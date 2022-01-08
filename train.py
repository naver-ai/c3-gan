import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from datasets import Dataset
from utils import *
from evals import *
from inception import InceptionV3



class Trainer(object):
    def __init__(self):
        ## define hyperparameters
        self.bs = cfg.BATCH_SIZE
        self.num_gt_cls = cfg.NUM_GT_CLASSES
        self.over = cfg.OVER
        self.num_cls = self.num_gt_cls*self.over
        self.temp = cfg.TEMP
        if cfg.PERT == 'w':
            self.min_s, self.max_s = 0.9, 1.1
            self.min_r, self.max_r = -2, 2
            self.min_t, self.max_t = -0.08, 0.08
        else:
            self.min_s, self.max_s = 0.8, 1.5
            self.min_r, self.max_r = -15, 15
            self.min_t, self.max_t = -0.15, 0.15
        try: 
            os.makedirs(cfg.SAVE_DIR)
        except:
            pass
        self.summary_writer = SummaryWriter(log_dir=cfg.SAVE_DIR)

        ## define dataloader
        self.dataset = Dataset(cfg.DATA_DIR)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.bs, drop_last=True, num_workers=8, shuffle=True, pin_memory=True)
        self.test_dataset = Dataset(cfg.DATA_DIR, 'test')
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.bs, drop_last=False, num_workers=8, shuffle=False)

        ## define models to train
        print('Generate model.')
        self.netG, self.netD = load_network(self.num_cls)   
        self.optimizerG, self.optimizerD = define_optimizers(self.netG, self.netD)
        self.CE = nn.CrossEntropyLoss()
        self.eye = torch.eye(self.num_cls).cuda()

        if cfg.MODEL_PATH != '':
            print('Load pre-trained model.')
            self.resume(cfg.MODEL_PATH)

        print('Get the statistic of training images for computing fid score.')
        self.inception = InceptionV3([3]).cuda()
        self.inception.eval()
        pred_arr = np.empty((len(self.dataset), 2048))
        start_idx = 0
        for data in self.dataloader:
            batch = data[0].cuda()
            with torch.no_grad():
                pred = self.inception(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
        self.mu = np.mean(pred_arr, axis=0)
        self.sig = np.cov(pred_arr, rowvar=False)


    def save(self, file_path):
        state = {'netG' : self.netG.state_dict(), 'netD' : self.netD.state_dict()}
        torch.save(state, file_path)


    def resume(self, model_path):
        state = torch.load(model_path)
        self.netG.load_state_dict(state['netG'])
        self.netD.load_state_dict(state['netD'])

 
    def prepare_code(self):
        rand_z = torch.FloatTensor(self.bs, cfg.GAN.Z_DIM).normal_(0, 1).cuda()
        rand_cz = torch.FloatTensor(self.bs, cfg.GAN.CZ_DIM).normal_(0, 1).cuda()
        rand_c = torch.zeros(self.bs, self.num_cls).cuda()
        rand_idx = [i for i in range(self.num_cls)]
        random.shuffle(rand_idx)
        for i, idx in enumerate(rand_idx[:self.bs]):
            rand_c[i, idx] = 1
        return rand_z, rand_cz, rand_c
    

    def prepare_data(self, data):
        real_img, aug_img = data 
        real_img = real_img.cuda()
        aug_img = aug_img.cuda()
        return  real_img, aug_img


    def random_grid(self, size):
        rand_s = torch.FloatTensor(self.bs, 1).uniform_(self.min_s, self.max_s)
        rand_r = torch.FloatTensor(self.bs, 1).uniform_(self.min_r, self.max_r)
        rand_tx = torch.FloatTensor(self.bs, 1).uniform_(self.min_t, self.max_t)
        rand_ty = torch.FloatTensor(self.bs, 1).uniform_(self.min_t, self.max_t)
        theta = [[[rand_s[i]*torch.cos(to_rad(rand_r[i])), -rand_s[i]*torch.sin(to_rad(rand_r[i])), rand_tx[i]],
                  [rand_s[i]*torch.sin(to_rad(rand_r[i])), rand_s[i]*torch.cos(to_rad(rand_r[i])), rand_ty[i]]]
                  for i in range(self.bs)]
        theta = torch.tensor(theta)
        grid = F.affine_grid(theta, size, align_corners=True).cuda()
        return grid


    def train_D(self):
        self.optimizerD.zero_grad()

        ## forward pass
        real_info, real_adv, _ = self.netD(self.real_img, self.eye)
        fake_info, fake_adv, class_emb, fg_fake_info = \
                            self.netD(self.fake_img.detach(), self.eye, self.fg_mask.detach()*self.fake_img.detach())

        ## adversarial loss
        real_adv_loss = torch.nn.ReLU()(1.0 - real_adv).mean()
        fake_adv_loss = torch.nn.ReLU()(1.0 + fake_adv).mean()

        ## info loss & augmented info loss
        f = F.normalize(fake_info, p=2, dim=1)
        ff = F.normalize(fg_fake_info, p=2, dim=1)
        c = F.normalize(class_emb, p=2, dim=1)

        class_dist = torch.cat([torch.matmul(f, c[i]).unsqueeze(-1)/self.temp for i in range(self.num_cls)], 1)
        info_loss = 5*self.CE(class_dist, torch.argmax(self.rand_c, 1))
        class_dist_aug = torch.cat([torch.matmul(ff, c[i]).unsqueeze(-1)/self.temp for i in range(self.num_cls)], 1)
        info_loss_aug = self.CE(class_dist_aug, torch.argmax(self.rand_c, 1))

        ## real image contrastive loss
        labels = torch.cat([torch.arange(self.bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        real_aug_feat, _, _ = self.netD(self.aug_img, self.eye)
        feat = torch.cat([real_info, real_aug_feat], 0)
        feat = F.normalize(feat, p=2, dim=1)
        similarity_matrix = torch.matmul(feat, feat.T)

        mask = torch.eye(feat.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits = logits/self.temp

        contrastive_loss = self.CE(logits, labels)

        ## entropy regularizations
        rr = F.normalize(real_info, p=2, dim=1)
        class_dist_real = torch.cat([torch.matmul(rr, c[i]).unsqueeze(-1)/self.temp for i in range(self.num_cls)], 1)
        class_dist_real = F.softmax(class_dist_real, 1)
        entropy_reg_1 = -0.1*(class_dist_real*torch.log(class_dist_real)).sum(1).mean()
        entropy_reg_2 = 0.1*(class_dist_real.mean(0)*torch.log(class_dist_real.mean(0))).sum()

        ## backward pass
        D_loss = real_adv_loss + fake_adv_loss + info_loss + info_loss_aug + contrastive_loss\
                 + entropy_reg_1 + entropy_reg_2
        D_loss.backward()
        self.optimizerD.step()

        ## log training losses
        if self.steps % 500 == 0:
            self.summary_writer.add_scalar('D/real_loss', real_adv_loss.item(), self.steps)
            self.summary_writer.add_scalar('D/fake_loss', fake_adv_loss.item(), self.steps)
            self.summary_writer.add_scalar('D/info_loss', info_loss.item(), self.steps)
            self.summary_writer.add_scalar('D/info_loss_aug', info_loss_aug.item(), self.steps)
            self.summary_writer.add_scalar('D/contrastive_loss', contrastive_loss.item(), self.steps)
            self.summary_writer.add_scalar('D/entropy_reg_1', entropy_reg_1.item(), self.steps)
            self.summary_writer.add_scalar('D/entropy_reg_2', entropy_reg_2.item(), self.steps)


    def train_G(self):
        self.optimizerG.zero_grad()

        ## forward pass
        fake_info, fake_adv, class_emb, fg_fake_info = self.netD(self.fake_img, self.eye, self.fg_mask*self.fake_img)

        ## adversarial loss
        adv_loss = -fake_adv.mean()

        ## info loss & augmented info loss
        f = F.normalize(fake_info, p=2, dim=1)
        ff = F.normalize(fg_fake_info, p=2, dim=1)
        c = F.normalize(class_emb, p=2, dim=1)

        class_dist = torch.cat([torch.matmul(f, c[i]).unsqueeze(-1)/self.temp for i in range(self.num_cls)], 1)
        info_loss = 5*self.CE(class_dist, torch.argmax(self.rand_c, 1))
        class_dist_aug = torch.cat([torch.matmul(ff, c[i]).unsqueeze(-1)/self.temp for i in range(self.num_cls)], 1)
        info_loss_aug = self.CE(class_dist_aug, torch.argmax(self.rand_c, 1))

        ## mask regularizations
        mean_map = torch.mean(self.fg_mask.view(self.bs, -1), 1)
        mask_reg_loss = torch.max(torch.zeros_like(mean_map), 0.1-mean_map).mean() +\
                        torch.max(torch.zeros_like(mean_map), mean_map-0.9).mean() +\
                        binary_entropy(self.fg_mask).mean()

        ## backward pass
        G_loss =  adv_loss + info_loss + info_loss_aug + mask_reg_loss
        G_loss.backward()
        self.optimizerG.step()

        ## log training losses
        if self.steps % 500 == 0:
            self.summary_writer.add_scalar('G/adv_loss', adv_loss.item(), self.steps)
            self.summary_writer.add_scalar('G/info_loss', info_loss.item(), self.steps)
            self.summary_writer.add_scalar('G/info_loss_aug', info_loss_aug.item(), self.steps)
            self.summary_writer.add_scalar('G/mask_reg_loss', mask_reg_loss.item(), self.steps)


    def train(self):

        self.steps = 0
        self.max_acc = 0
        self.max_nmi = 0
        self.min_fid = 999

        print("Start training.")
        for epoch in range(cfg.MAX_EPOCH):              

            for data in self.dataloader: 
            
                ## update D
                self.real_img, self.aug_img = self.prepare_data(data)
                rand_z, rand_cz, self.rand_c = self.prepare_code()
                _, self.fg_mask, self.fg_img, self.fake_img = self.netG(rand_z, rand_cz, self.rand_c, self.random_grid(self.real_img.size()))
                self.train_D()

                ## update G
                rand_z, rand_cz, self.rand_c = self.prepare_code()
                _, self.fg_mask, self.fg_img, self.fake_img = self.netG(rand_z, rand_cz, self.rand_c, self.random_grid(self.real_img.size()))
                self.train_G()

                self.steps += 1

            ## evaluate every x epochs
            if epoch % cfg.EVAL_INTERVAL == 0:
                ## switch to eval mode
                self.netG.eval()   
                self.netD.eval()  

                with torch.no_grad():
                    rand_z, rand_cz, rand_c = self.prepare_code()
                    rand_z = torch.cat([rand_z[:1].repeat(4, 1), rand_z[1:5]], 0)
                    rand_cz = torch.cat([rand_cz[:1].repeat(4, 1), rand_cz[1:5]], 0)
                    rand_c = torch.cat([rand_c[:4], rand_c[4:5].repeat(4, 1)], 0)

                    self.bg_img, self.fg_mask, self.fg_img, self.fake_img = self.netG(rand_z, rand_cz, rand_c)
                    self.bg_img = postprocess(self.bg_img)
                    self.fg_mask = postprocess(self.fg_mask.repeat(1,3,1,1))
                    self.fg_img = postprocess(self.fg_img)
                    self.fake_img = postprocess(self.fake_img)

                    vis = torch.cat([self.bg_img, self.fg_mask, self.fg_img, self.fake_img])
                    vis = vutils.make_grid(vis, nrow=8, padding=10, pad_value=1)
                    self.summary_writer.add_image('Image_sample', vis, self.steps)

                ## get acc and nmi scores on predictions
                pred_c = []
                real_c = []
                with torch.no_grad():   
                    for img, lab in self.test_dataloader:
                        real_img = img.cuda()
                        feat, _, class_emb = self.netD(real_img, self.eye)
                        f = F.normalize(feat, p=2, dim=1)
                        c = F.normalize(class_emb, p=2, dim=1)
                        class_dist = torch.cat([torch.matmul(f, c[i]).unsqueeze(-1)/self.temp for i in range(self.num_cls)], 1)
                        pred_c += list(torch.argmax(class_dist, 1).cpu().numpy())
                        real_c += list(lab.cpu().numpy())
                c_table = get_stat(pred_c, self.num_cls, real_c)

                for i in range(1,self.over):
                    c_table[self.num_cls//self.over*i:self.num_cls//self.over*(i+1),:] = c_table[:self.num_cls//self.over,:]
                idx_map = get_match(c_table)
                cur_acc = get_acc(c_table, idx_map, self.over)
                cur_nmi = get_nmi(c_table[:self.num_gt_cls,:])

                ## get fid score on randomly generated samples
                pred_arr = np.empty(((len(self.dataset)//self.bs)*self.bs, 2048))
                start_idx = 0
                for i in range(len(self.dataset)//self.bs):
                    rand_z, rand_cz, rand_c = self.prepare_code()
                    with torch.no_grad():
                        _, _, _, fake_img = self.netG(rand_z, rand_cz, rand_c)
                        pred = self.inception(fake_img)[0]
                    if pred.size(2) != 1 or pred.size(3) != 1:
                        print('size mismatch error occurred during the fid score computation!')
                        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
                    pred_arr[start_idx:start_idx + pred.shape[0]] = pred
                    start_idx = start_idx + pred.shape[0]
                cur_mu = np.mean(pred_arr, axis=0)
                cur_sig = np.cov(pred_arr, rowvar=False)
                cur_fid = calculate_frechet_distance(self.mu, self.sig, cur_mu, cur_sig)
                
                print(str(epoch)+"th epoch finished", "\t fid : ", "{:.3f}".format(cur_fid),\
                                                      "\t acc : ", "{:.3f}".format(cur_acc),\
                                                      "\t nmi : ", "{:.3f}".format(cur_nmi))

                ## log of evaluation scores
                self.summary_writer.add_scalar('test/ACC', cur_acc, self.steps)
                self.summary_writer.add_scalar('test/NMI', cur_nmi, self.steps)
                self.summary_writer.add_scalar('test/FID', cur_fid, self.steps)

                ## save optimal versions of model in terms of fid and acc scores
                if cur_fid < self.min_fid:
                    self.min_fid = cur_fid
                    self.save(os.path.join(cfg.SAVE_DIR, 'best_fid.pt'))
                if cur_acc > self.max_acc:
                    self.max_acc = cur_acc
                    self.save(os.path.join(cfg.SAVE_DIR, 'best_acc.pt'))

                ## switch to training mode
                self.netG.train()
                self.netD.train()

        print("Training completed.")


if __name__ == "__main__":

    trainer = Trainer()
    trainer.train()
