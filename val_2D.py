import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if (pred.sum() > 0) and (gt.sum() > 0):
        
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        
        return dice, hd95
    
    elif (pred.sum() == 0) and (gt.sum() == 0):
        return 1,0
    return 0, 1


def test_single_volume_tdm(image_pre, image, label,net, classes, patch_size=[256, 256], is_TDM=True):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     prediction = np.zeros_like(label)
    image_pre = image_pre.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros((label.shape[0],patch_size[0],patch_size[1]))
    true_label = np.zeros((label.shape[0],patch_size[0],patch_size[1]))
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        slice_pre = image_pre[ind,:,:]
        slice_label = label[ind,:,:]
        x, y = slice.shape[0], slice.shape[1]
        
        label_l = slice_label[:112,:112]
        label_r = slice_label[-112:,-112:]
        
        select_lr = 0
        if label_l.sum() < label_r.sum():
            select_lr = 1
        
        if select_lr == 0:
            slice_pre = slice_pre[:112,:112]
            slice = slice[:112,:112]
            slice_label = slice_label[:112,:112]
        else :
            slice_pre = slice_pre[-112:,-112:]
            slice = slice[-112:,-112:]
            slice_label = slice_label[-112:,-112:]
        
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        slice_pre = torch.from_numpy(slice_pre).unsqueeze(
            0).unsqueeze(0).float().cuda()
        
        tdm = torch.ones(input.shape).cuda()
        tdm_batch = (input+1.0) - slice_pre
        tdm = torch.cat((tdm,tdm_batch), dim=1)
        
        
        true_label[ind] = slice_label
        net.eval()
        with torch.no_grad():
            if is_TDM == True:
                out = torch.argmax(torch.softmax(
                    net(input,tdm,istrain=False), dim=1), dim=1).squeeze(0)
            elif is_TDM == False:
                out = torch.argmax(torch.softmax(
                    net(input,istrain=False), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            pred = out
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, true_label == i))
        
        
    return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
#     prediction = np.zeros_like(label)
    prediction = np.zeros((label.shape[0],patch_size[0],patch_size[1]))
    true_label = np.zeros((label.shape[0],patch_size[0],patch_size[1]))
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice_label = label[ind,:,:]
        
        
        label_l = slice_label[:112,:112]
        label_r = slice_label[-112:,-112:]
        
        select_lr = 0
        if label_l.sum() < label_r.sum():
            select_lr = 1
        
        if select_lr == 0:
            slice = slice[:112,:112]
            slice_label = slice_label[:112,:112]
        else :
            slice = slice[-112:,-112:]
            slice_label = slice_label[-112:,-112:]
        
        
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        
        true_label[ind] = slice_label
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            pred = out
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, true_label == i))
     
        
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
