import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import glob
# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# parser.add_argument('--tdm', type=str, default='False',
#                     help='using tdm')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    if (pred.sum() > 0) and (gt.sum() > 0):
        
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        
        return dice, asd, hd95
    
    elif (pred.sum() == 0) and (gt.sum() == 0):
        return 1,0,0
    return 0, 1, 1


def test_single_volume_tdm(case, net, test_save_path, FLAGS):
    
    val_h5f_list = glob.glob(os.path.join('/data/UFMR_DCMR_h5py', '{}_slice_*').format(case))
    val_volume = h5py.File(val_h5f_list[0], 'r')
    
    val_volume_image_pre = val_volume['ufmr_1'][:]
    val_volume_image = val_volume['ufmr_last'][:]
    val_volume_label = val_volume['label'][:]
    
    prediction = np.zeros_like(val_volume_label)
        
    for val_slice_num in range(1, len(val_h5f_list)):
        h5f = h5py.File(val_h5f_list[val_slice_num], 'r')
        image_pre = h5f['ufmr_1'][:]
        image = h5f['ufmr_last'][:]
        label = h5f['label'][:]

        val_volume_image_pre = np.concatenate((val_volume_image_pre,image_pre), axis=0)
        val_volume_image = np.concatenate((val_volume_image,image), axis=0)
        val_volume_label = np.concatenate((val_volume_label,label), axis=0)
    image_pre = val_volume_image_pre
    image = val_volume_image
    label = val_volume_label   
    
    prediction = np.zeros((label.shape[0],112,112))
    true_label = np.zeros((label.shape[0],112,112))
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        slice_pre = image_pre[ind,:,:]
        x, y = slice.shape[0], slice.shape[1]
        
        slice = slice[-112:, -112:]
        slice_pre = slice_pre[-112:, -112:]
        
        
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        
        slice_pre = torch.from_numpy(slice_pre).unsqueeze(
            0).unsqueeze(0).float().cuda()
        
        tdm = torch.ones(input.shape).cuda()
        tdm_batch = (input+1.0) - slice_pre
        tdm = torch.cat((tdm,tdm_batch), dim=0)
        
        true_label[ind] = label[ind,-112:,-112:]
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input,tdm,istrain=False)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
            pred = out
            prediction[ind] = pred
    first_metric1, second_metric1, third_metric1 = calculate_metric_percase(prediction == 1, true_label == 1)
#     second_metric1 = calculate_metric_percase(prediction == 1, true_label == 1)
#     third_metric1 = calculate_metric_percase(prediction == 1, true_label == 1)
    
    
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(true_label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred_r.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img_r.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt_r.nii.gz")
    
    prediction = np.zeros((label.shape[0],112,112))
    true_label = np.zeros((label.shape[0],112,112))
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        slice_pre = image_pre[ind,:,:]
        x, y = slice.shape[0], slice.shape[1]
        
        slice = slice[:112, :112]
        slice_pre = slice_pre[:112, :112]
        
        
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        
        slice_pre = torch.from_numpy(slice_pre).unsqueeze(
            0).unsqueeze(0).float().cuda()
        
        tdm = torch.ones(input.shape).cuda()
        tdm_batch = (input+1.0) - slice_pre
        tdm = torch.cat((tdm,tdm_batch), dim=0)
        
        true_label[ind] = label[ind,:112, :112]
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input,tdm,istrain=False)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
            pred = out
            prediction[ind] = pred
    first_metric2, second_metric2, third_metric2 = calculate_metric_percase(prediction == 1, true_label == 1)
#     second_metric2 = calculate_metric_percase(prediction == 1, label == 1)
#     third_metric2 = calculate_metric_percase(prediction == 1, label == 1)
    
    first_metric = (first_metric1+first_metric2)/2
    second_metric = (second_metric1+second_metric2)/2
    third_metric = (third_metric1+third_metric2)/2
    
   
    
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(true_label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred_l.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img_l.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt_l.nii.gz")
    
    
    return first_metric, second_metric, third_metric



def test_single_volume(case, net, test_save_path, FLAGS):
    
    val_h5f_list = glob.glob(os.path.join('/data/UFMR_DCMR_h5py', '{}_slice_*').format(case))
    val_volume = h5py.File(val_h5f_list[0], 'r')
    
    val_volume_image_pre = val_volume['ufmr_1'][:]
    val_volume_image = val_volume['ufmr_last'][:]
    val_volume_label = val_volume['label'][:]
    
    prediction = np.zeros_like(val_volume_label)
        
    for val_slice_num in range(1, len(val_h5f_list)):
        h5f = h5py.File(val_h5f_list[val_slice_num], 'r')
        image_pre = h5f['ufmr_1'][:]
        image = h5f['ufmr_last'][:]
        label = h5f['label'][:]

        val_volume_image_pre = np.concatenate((val_volume_image_pre,image_pre), axis=0)
        val_volume_image = np.concatenate((val_volume_image,image), axis=0)
        val_volume_label = np.concatenate((val_volume_label,label), axis=0)
    image_pre = val_volume_image_pre
    image = val_volume_image
    label = val_volume_label   
    
    prediction = np.zeros((label.shape[0],112,112))
    true_label = np.zeros((label.shape[0],112,112))
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = slice[-112:, -112:]
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        true_label[ind] = label[ind,-112:,-112:]
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
            pred = out
            prediction[ind] = pred
    first_metric1, second_metric1, third_metric1 = calculate_metric_percase(prediction == 1, true_label == 1)
#     second_metric1 = calculate_metric_percase(prediction == 1, true_label == 1)
#     third_metric1 = calculate_metric_percase(prediction == 1, true_label == 1)
    
    
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(true_label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred_r.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img_r.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt_r.nii.gz")
    
    prediction = np.zeros((label.shape[0],112,112))
    true_label = np.zeros((label.shape[0],112,112))
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = slice[:112, :112]
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        
        true_label[ind] = label[ind,:112, :112]
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
#             pred = zoom(out, (x / 256, y / 256), order=0)
            pred = out
            prediction[ind] = pred
    first_metric2, second_metric2, third_metric2 = calculate_metric_percase(prediction == 1, true_label == 1)
#     second_metric2 = calculate_metric_percase(prediction == 1, label == 1)
#     third_metric2 = calculate_metric_percase(prediction == 1, label == 1)
    
    first_metric = (first_metric1+first_metric2)/2
    second_metric = (second_metric1+second_metric2)/2
    third_metric = (third_metric1+third_metric2)/2
    
   
    
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(true_label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred_l.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img_l.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt_l.nii.gz")
    
    
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
#     save_mode_path = os.path.join(
#         snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
    save_mode_path = os.path.join(
        snapshot_path, 'model1_iter_24000.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume_tdm(case, net, test_save_path, FLAGS)
#         if FLAGS.tdm == 'False':
#             print('test not tdm')
#             first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
#         elif FLAGS.tdm == 'True':
#             print('test tdm')
#             first_metric, second_metric, third_metric = test_single_volume_tdm(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    print(FLAGS)
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
