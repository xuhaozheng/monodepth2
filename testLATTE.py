import argparse
import time
import torch
import numpy as np
import torch.optim as optim

# custom modules

from loss import MonodepthLoss
from utils import get_model, transformer_model, to_device, prepare_dataloader, readlines
from skimage.metrics import structural_similarity as ssim
from loss import MonodepthLoss, ICPLoss
import os
import torch.nn.functional as F
import PIL.Image as pil
from torchvision import transforms
import cv2
import matplotlib.cm as cm

# plot params

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15, 10)


def test(self):
    self.model.eval()
    # disparities = np.zeros((self.n_img,
    #                        self.input_height, self.input_width),
    #                        dtype=np.float32)
    # disparities_pp = np.zeros((self.n_img,
    #                           self.input_height, self.input_width),
    #                           dtype=np.float32)
    disparities = np.zeros((self.n_img,
                            self.args.full_height, self.args.full_width),
                           dtype=np.float32)
    disparities_pp = np.zeros((self.n_img,
                               self.args.full_height, self.args.full_width),
                              dtype=np.float32)

    if self.args.calculate_MAE:
        img_idx = 0
        errors = []
        ABE = []
        SCALE = []
        baseline = 4.2    # 5.045158438885819
        focal = 1135    # # 7866.0520212773545
        transform_resize = transforms.Resize((256, 320))
        with torch.no_grad():
            ground_truth_dir = os.path.join(file_dir, "Data/test/depth")
            test_data_dir = os.path.join(file_dir, "Data/test/image")
            image02_file = os.path.join(ground_truth_dir, 'image_02')
            image03_file = os.path.join(ground_truth_dir, 'image_03')
            for image in sorted(os.listdir(image02_file)):
                ground_truth_image_file_left = os.path.join(image02_file, image)
                ground_truth_image_file_right = os.path.join(image03_file, image)
                test_RGB_image_file_left = os.path.join(test_data_dir, 'image_02', image)
                test_RGB_image_file_right = os.path.join(test_data_dir, 'image_03', image)

                if not os.path.exists(ground_truth_image_file_left) and os.path.exists(
                        test_RGB_image_file_left):
                    print('Error: point could not found - {}'.format(test_RGB_image_file_left))
                ''' Load in Input image '''
                left_input_image = pil.open(test_RGB_image_file_left).convert('RGB')
                left_input_image = transform_resize(left_input_image)
                right_input_image = pil.open(test_RGB_image_file_right).convert('RGB')
                left_input_image = transforms.ToTensor()(left_input_image).unsqueeze(0)
                right_input_image = transforms.ToTensor()(right_input_image).unsqueeze(0)
                left_input_image = left_input_image.to(self.device)
                right_input_image = right_input_image.to(self.device)
                ''' Load in grond truth'''
                depth_gt_left = pil.open(ground_truth_image_file_left).convert('L')
                depth_gt_left = np.asarray(depth_gt_left, dtype="float32")

                

                disps = self.model(left_input_image)
                disps_upsample = F.interpolate(disps[0][:, 0, :, :].unsqueeze(1),
                                               [self.args.full_height, self.args.full_width], mode="bilinear",
                                               align_corners=False).squeeze().cpu().detach().numpy()
                depth_pred = (baseline * focal) / (disps_upsample*1280)
                # print('max depth_pred', np.sum(depth_pred>300))
                # print('depth_gt_left', depth_gt_left)
                mu_depth = depth_pred.mean()
                mu_gt = depth_gt_left.mean()
                scale = mu_gt / mu_depth
                SCALE.append(scale)

                difference_image = np.abs(depth_pred - depth_gt_left)
                non_valid_mask = np.logical_or(depth_gt_left < 25,
                                               depth_gt_left > 300)
                # non_valid_mask = np.logical_or(depth_gt_left < 2, depth_gt_left > 400)
                abe = (np.ma.array(difference_image, mask=non_valid_mask)).mean()
                ''''''#Visualization
                # depth_pred[depth_pred > 300] = 0
                # depth_pred[depth_pred < 25] = 0
                # save pred
                vmax_pred = np.percentile(depth_pred, 95)
                normalizer_pred = mpl.colors.Normalize(vmin=depth_pred.min(), vmax=vmax_pred)
                mapper_pred = cm.ScalarMappable(norm=normalizer_pred, cmap='magma')
                colormapped_im_pred = (mapper_pred.to_rgba(depth_pred)[:, :, :3]*255).astype(np.uint8)
                im_pred = pil.fromarray(colormapped_im_pred)
                dest_pred = os.path.join(file_dir, 'vis/{}_pred.jpeg'.format(img_idx))
                im_pred.save(dest_pred)
                #save gt
                vmax_gt = np.percentile(depth_pred, 95)
                normalizer_gt = mpl.colors.Normalize(vmin=depth_pred.min(), vmax=vmax_pred)
                mapper_gt = cm.ScalarMappable(norm=normalizer_gt, cmap='magma')
                colormapped_im_gt = (mapper_gt.to_rgba(depth_gt_left)[:, :, :3] * 255).astype(np.uint8)
                im_gt = pil.fromarray(colormapped_im_gt)
                dest_gt = os.path.join(file_dir, 'vis/{}_gt.jpeg'.format(img_idx))
                im_gt.save(dest_gt)

                # fig, axes = plt.subplots(nrows=2, ncols=1)
                # plt.subplot(2, 1, 1)
                # plt.imshow(depth_pred)
                # plt.title('Prediction')
                # plt.subplot(2, 1, 2)
                # plt.imshow(depth_gt_left)
                # plt.title('gt')
                # plt.savefig(os.path.join(file_dir, "vis/") + str(img_idx) + '.png')
                img_idx += 1

                ABE.append(abe)
                errors.append(compute_errors(depth_gt_left, depth_pred))
                mean_errors = np.array(errors).mean(0)

            print('Mean Absolute Error:', np.mean(ABE))
            print('std of Absolute Error:', np.std(ABE))
            print('mean scale:', np.mean(SCALE))
            print('std of scale:', np.std(SCALE))
            print('Finished Calculating Absolute Error')
            #### 7 criteria ####
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
            print("\n-> Done!")


def compute_errors(gt, pred, MIN_DEPTH=25, MAX_DEPTH=300):
    """Computation of error metrics between predicted and ground truth depths
    """
    mask = np.logical_and(gt >= MIN_DEPTH,  gt <= MAX_DEPTH)
    gt = gt[mask]
    pred = pred[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def main():
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()


if __name__ == '__main__':
    main()

