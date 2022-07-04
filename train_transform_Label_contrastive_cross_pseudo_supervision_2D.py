import argparse
import logging
import os
import random
import shutil
import sys
import time

import wandb
import SimpleITK as sitk
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

# from networks.unet import TripletLoss

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume_tdm

from ast import literal_eval

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/notebook/SSL4MIS/data/UFMR_DCMR', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='UFMR_DCMR/Cross_Pseudo_Supervision', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='Label_contrastive_UNet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--fold', type=int, default=0, help='fold')
args = parser.parse_args()

class PatchNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.nce_t = 0.07
    def forward(self, anc, pos, neg):
        bs = anc.shape[0]
        l_pos = torch.bmm(anc.view(bs,1,-1), pos.view(bs,-1,1))
        l_neg = torch.bmm(anc.view(bs,1,-1), neg.view(bs,-1,1))
        
        out = torch.cat((l_pos.view(bs,1), l_neg.view(bs,1)), dim=1)/self.nce_t
        
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=anc.device))
        
        return loss.mean()
    
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, ismlp):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ismlp = ismlp
    def forward(self, anchor, positive, negative):
        
        if self.ismlp == True: #batch*512

#             distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
#             distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
            cos_sim =  torch.nn.CosineSimilarity(dim=1)
            
            
            distance_positive = cos_sim(anchor,positive)
            distance_negative = cos_sim(anchor,negative)

#             if label == 'pos':
#                 losses = 

#             losses = 
            losses = F.relu(distance_positive - distance_negative + self.margin)
            
        elif self.ismlp ==False: #batch*256*7*7    
#             anchor, positive, negative = anchor.flatten(1), positive.flatten(1), negative.flatten(1)
#             distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
#             distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

#             cos_sim =  torch.nn.CosineSimilarity(dim=1)


#             distance_positive = cos_sim(anchor,positive)
#             distance_negative = cos_sim(anchor,negative)

            distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
            distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        
        
            losses = F.relu(distance_positive - distance_negative + self.margin)

#         print('losses:', losses)
        anchor_sum = anchor.sum(1)
        positive_sum = positive.sum(1)
        negative_sum = negative.sum(1)

        anchor_zero_idx = np.where(anchor_sum.float().cpu() == 0.0)
        positive_zero_idx = np.where(positive_sum.float().cpu() == 0.0)
        negative_zero_idx = np.where(negative_sum.float().cpu() == 0.0)

        losses[anchor_zero_idx] = losses[anchor_zero_idx]*0
        losses[positive_zero_idx] = losses[positive_zero_idx]*0
        losses[negative_zero_idx] = losses[negative_zero_idx]*0

        len_zero_losses = len(np.where(losses.cpu() == 0)[0])

        loss_size = len(losses) - len_zero_losses
        if loss_size == 0:
            #loss_size = 1
            return losses.sum()+self.margin
        return losses.sum()/loss_size

    

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def patients_to_slices(dataset, patiens_num, fold):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "UFMR_DCMR" in dataset:
        file = open('/notebook/SSL4MIS/data/UFMR_DCMR/ref_dict.list', "r")
        full_ref_dict=[]
        while True:
            line = file.readline()
            if not line:
                break
            full_ref_dict.append(line)

        file.close()
        ref_dict = literal_eval(full_ref_dict[fold])
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    ismlp = False

    def create_model(ema=False):
        # Network definition
        print(args.model)
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()
    
    wandb.watch(model1)
    wandb.watch(model2)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size, use_tdm=True)
    ]), fold=str(args.fold)+'_fold_')
    
    db_val = BaseDataSets(base_dir=args.root_path, split="val", fold=str(args.fold)+'_fold_')

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num, args.fold)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

#     contrastive_loss = TripletLoss(margin=1.0,ismlp=ismlp)
    contrastive_loss = ContrastiveLoss()
#     writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    
    best_performance_combine = 0.0
    
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, (sampled_batch, sampled_batch2) in enumerate(trainloader):

            volume_pre_batch, volume_batch, label_batch = sampled_batch['image_pre'], sampled_batch['image'], sampled_batch['label']
            volume_pre_batch, volume_batch, label_batch = volume_pre_batch.cuda(), volume_batch.cuda(), label_batch.cuda()
            
            volume_pre_batch2, volume_batch2, label_batch2 = sampled_batch2['image_pre'], sampled_batch2['image'], sampled_batch2['label']
            volume_pre_batch2, volume_batch2, label_batch2 = volume_pre_batch2.cuda(), volume_batch2.cuda(), label_batch2.cuda()
#             print('volume_batch:',volume_batch.shape) torch.Size([24, 1, 112, 112])
#             print('label_batch:',label_batch.shape) torch.Size([24, 112, 112])
#             print('label_batch[:][:args.labeled_bs]:',label_batch[:][:args.labeled_bs].shape) torch.Size([12, 112, 112])
#             print('label_batch[:args.labeled_bs]:',label_batch[:args.labeled_bs].shape) torch.Size([12, 112, 112])
#             print('label_batch[:args.labeled_bs].unsqueeze(1)',label_batch[:args.labeled_bs].unsqueeze(1).shape) torch.Size([12, 1, 112, 112])
            tdm = torch.ones(volume_batch.shape).cuda()
            tdm_batch = (volume_batch+1.0) - volume_pre_batch
            tdm = torch.cat((tdm,tdm_batch), dim=1)
            
            tdm2 = torch.ones(volume_batch2.shape).cuda()
            tdm_batch2 = (volume_batch2+1.0) - volume_pre_batch2
            tdm2 = torch.cat((tdm2,tdm_batch2), dim=1)
            
            
            outputs1, positive_feature1, negative_feature1 = model1(volume_batch,label_batch,args.labeled_bs,istrain=True,ismlp=ismlp)
            outputs_soft1 = torch.softmax(outputs1*tdm, dim=1)
            #outputs_soft1 = outputs_soft1*tdm
            
            outputs12, positive_feature12, negative_feature12 = model1(volume_batch2,label_batch2,args.labeled_bs,istrain=True,ismlp=ismlp)
            outputs_soft12 = torch.softmax(outputs12, dim=1)
            #outputs_soft12 = outputs_soft12

          
            #version_1
    
            outputs2, positive_feature2, negative_feature2 = model2(volume_batch2,label_batch2,args.labeled_bs,istrain=True,ismlp=ismlp) 
            outputs_soft2 = torch.softmax(outputs2*tdm2, dim=1)
            #outputs_soft2 = outputs_soft2*tdm2
            
            outputs21, positive_feature21, negative_feature21 = model2(volume_batch,label_batch,args.labeled_bs,istrain=True,ismlp=ismlp) 
            outputs_soft21 = torch.softmax(outputs21, dim=1)
            #outputs_soft21 = outputs_soft21*tdm


#             outputs2, positive_feature2, negative_feature2 = model2((tdm_batch-1.0),tdm,istrain=True,ismlp=ismlp) #version_2
#             outputs_soft2 = torch.softmax(outputs2, dim=1)
#             outputs_soft2 = outputs_soft2

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch2[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch2[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(outputs_soft21[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft12[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs1)
            pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs2)

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2
            
            contrastive_loss1_p = contrastive_loss(positive_feature1, positive_feature2, negative_feature2)
            contrastive_loss1_n = contrastive_loss(negative_feature1, negative_feature2, positive_feature2)
            contrastive_loss2_p = contrastive_loss(positive_feature2, positive_feature1, negative_feature1)
            contrastive_loss2_n = contrastive_loss(negative_feature2, negative_feature1, positive_feature1)

#             print('contrastive_loss1_p:',contrastive_loss1_p)
#             print('contrastive_loss1_n:',contrastive_loss1_n)

#             print('contrastive_loss2_p:',contrastive_loss2_p)
#             print('contrastive_loss2_n:',contrastive_loss2_n)


            final_contrastive_loss = (contrastive_loss1_p+contrastive_loss1_n+contrastive_loss2_p+contrastive_loss2_n)/4
            #print(final_contrastive_loss)
            loss = model1_loss + model2_loss + consistency_weight*final_contrastive_loss


            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

                
            wandb.log({
                "iter_num" : iter_num,
                "lr" : lr_,
                "model1_loss" : model1_loss,
                "model2_loss" : model2_loss,
                "consistency_weight" : consistency_weight,
                "sum_contrastive_loss" : final_contrastive_loss,
            })    
#             writer.add_scalar('lr', lr_, iter_num)
#             writer.add_scalar(
#                 'consistency_weight/consistency_weight', consistency_weight, iter_num)
#             writer.add_scalar('loss/model1_loss',
#                               model1_loss, iter_num)
#             writer.add_scalar('loss/model2_loss',
#                               model2_loss, iter_num)
            #############logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))
            
            if iter_num % 50 == 0:
               
                
                image = volume_batch[1, 0:1, :, :]
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                outputs = outputs[1, ...] * 50
                outputs2 = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                outputs2 = outputs2[1, ...] * 50
                labs = label_batch[1, ...].unsqueeze(0) * 50
                
                
                source = sitk.GetImageFromArray(image.data.cpu())
                sitk.WriteImage(source, os.path.join('./'+args.exp, '{}_image.nii.gz'.format(iter_num)))
        
                o = sitk.GetImageFromArray(outputs.data.cpu().float())
                sitk.WriteImage(o, os.path.join('./'+args.exp, '{}_o.nii.gz'.format(iter_num)))
                
                o2 = sitk.GetImageFromArray(outputs2.data.cpu().float())
                sitk.WriteImage(o2, os.path.join('./'+args.exp, '{}_o2.nii.gz'.format(iter_num)))
                
                l = sitk.GetImageFromArray(labs.data.cpu())
                sitk.WriteImage(l, os.path.join('./'+args.exp, '{}_l.nii.gz'.format(iter_num)))

#                 image = volume_batch[1, 0:1, :, :]
#                 writer.add_image('train/Image', image, iter_num)
#                 outputs = torch.argmax(torch.softmax(
#                     outputs1, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/model1_Prediction',
#                                  outputs[1, ...] * 50, iter_num)
#                 outputs = torch.argmax(torch.softmax(
#                     outputs2, dim=1), dim=1, keepdim=True)
#                 writer.add_image('train/model2_Prediction',
#                                  outputs[1, ...] * 50, iter_num)
#                 labs = label_batch[1, ...].unsqueeze(0) * 50
#                 writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_tdm(
                        sampled_batch['image_pre'], sampled_batch['image'], sampled_batch['label'], model1, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    wandb.log({
                        "v_iter_num" : iter_num,
                        "model1_val_{}_dice".format(class_i+1) : metric_list[class_i, 0],
                        "model1_val_{}_hd95".format(class_i+1) : metric_list[class_i, 1],
                    })

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                wandb.log({
                        "v_iter_num" : iter_num,
                        "model1_val_mean_dice" : performance1,
                        "model1_val_mean_hd95" : mean_hd951,
                    })
                
                combine = (performance1*10.0-mean_hd951+5)/10

                if combine > best_performance_combine:
                    best_performance_combine = combine
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(combine, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1_combine.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

#                 logging.info(
#                     'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_tdm(
                        sampled_batch['image_pre'], sampled_batch['image'], sampled_batch['label'], model2, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    wandb.log({
                        "v_iter_num" : iter_num,
                        "model2_val_{}_dice".format(class_i+1) : metric_list[class_i, 0],
                        "model2_val_{}_hd95".format(class_i+1) : metric_list[class_i, 1],
                    })

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                wandb.log({
                        "v_iter_num" : iter_num,
                        "model2_val_mean_dice" : performance2,
                        "model2_val_mean_hd95" : mean_hd952,
                    })

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
#     writer.close()


if __name__ == "__main__":
    
    wandb.init(project=args.exp)
    wandb.run.name = args.exp+str(args.labeled_num)
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists('./'+args.exp):
        os.makedirs('./'+args.exp)
        
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
