import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import argparse
import logging # Assuming setup_logger uses this
import datetime
from tqdm import tqdm
import wandb
import re # Keep for potential future use, but not used for epoch extraction now

sys.path.append('.')
from utils.SAR_loader import get_dataloader
from networks.vae import *
from utils.set import *
from networks_V1.vae_cbs import CVAE_SAR_PreActResNet, CVAE_SAR_Res, CVAE_SAR_vgg, CVAE_SAR_mobile, CVAE_SAR_densenet
from networks_V1.vae import CVAE_SAR_Wid

# ==============================================================================
# Assume these functions/classes are defined elsewhere or in this file:
# - get_dataloader(dataset, batch_size, size, train_path, test_path, mean, std)
# - CVAE_SAR_Res(d, z, num_classes) # The model architecture used for training
# - AverageMeter()
# - accuracy(output, target, topk)
# - total_correlation(hi, mu, logvar)
# - setup_logger(save_dir)
# - set_random_seed(seed)
# ==============================================================================

# --- Logging Individual Images Function (Removed epoch) ---
def log_individual_images(dataloader, model, args, batch_num=2, save_local=False):
    """
    Logs individual original, reconstructed, and residual images to wandb
    for a specified number of batches. Optionally saves them locally.
    (Epoch information removed).
    """
    SAR_dataloader = dataloader
    model.eval()

    local_save_dir = None
    if save_local:
        # Directory no longer includes epoch number
        local_save_dir = os.path.join(args.save_dir, "reconstructions")
        if not os.path.exists(local_save_dir):
            os.makedirs(local_save_dir, exist_ok=True)
            print(f"Created local save directory: {local_save_dir}")

    images_logged_count = 0

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(SAR_dataloader):
            if batch_idx >= batch_num:
                break

            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda().view(-1, )
            else:
                X, y = X, y.view(-1, )

            try:
                out, hi, gx, mu, logvar = model(X)
            except Exception as e:
                print(f"Error during model forward pass in batch {batch_idx}: {e}. Skipping.")
                continue

            current_batch_size = X.size(0)

            for i in range(current_batch_size):
                original_img = X[i]
                reconstructed_img = gx[i]
                residual_img = original_img - reconstructed_img

                # WandB key no longer includes epoch
                base_key = f"Test_Images/Batch_{batch_idx}_Index_{i}"
                try:
                    wandb.log({f"{base_key}/Original": wandb.Image(original_img)}, commit=False)
                    wandb.log({f"{base_key}/Reconstructed": wandb.Image(reconstructed_img)}, commit=False)
                    wandb.log({f"{base_key}/Residual": wandb.Image(residual_img)}, commit=False)
                except Exception as e:
                    print(f"Error logging image {base_key} to WandB: {e}")

                if save_local and local_save_dir:
                    # Filename no longer includes epoch implicitly via directory structure only
                    local_base_filename = f"batch{batch_idx}_idx{i}"
                    try:
                        torchvision.utils.save_image(original_img, os.path.join(local_save_dir, f"{local_base_filename}_orig.png"), normalize=True)
                        torchvision.utils.save_image(reconstructed_img, os.path.join(local_save_dir, f"{local_base_filename}_recon.png"), normalize=True)
                        torchvision.utils.save_image(residual_img, os.path.join(local_save_dir, f"{local_base_filename}_resid.png"), normalize=True)
                    except Exception as e:
                         print(f"Error saving local image {os.path.join(local_save_dir, local_base_filename)}: {e}")

                images_logged_count += 1

    print(f'Individual image logging complete for {images_logged_count} images across {min(batch_num, len(SAR_dataloader))} batches!')


# --- Test Function (Removed epoch parameter) ---
def test(model, testloader, args):
    """ Performs the testing loop and logs metrics/images (Epoch removed). """
    model.eval()
    acc_avg = AverageMeter()
    sparse_avg = AverageMeter()
    top1_res = AverageMeter()  # 残差图像分类准确率
    top1_orig = AverageMeter() # 原始图像分类准确率
    top1_recon = AverageMeter() # 重建图像分类准确率
    TC = AverageMeter()

    with torch.no_grad():
        # tqdm description no longer includes epoch
        pbar = tqdm(enumerate(testloader), total=len(testloader), desc="Testing")
        for batch_idx, (x, y) in pbar:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda().view(-1, )
            else:
                 x, y = x, y.view(-1, )

            bs = x.size(0)
            x_flatten = x.view(bs, -1)
            norm = torch.norm(x_flatten, p=2, dim=1)
            norm = torch.clamp(norm, min=1e-6).unsqueeze(1).unsqueeze(2).unsqueeze(3)

            try:
                out, hi, gx, mu, logvar = model(x)

                # 计算重建精度
                acc_gx = 1 - F.mse_loss(torch.div(gx, norm), torch.div(x, norm), reduction='sum') / bs
                acc_rx = 1 - F.mse_loss(torch.div(x - gx, norm), torch.div(x, norm), reduction='sum') / bs
                acc_avg.update(acc_gx.item(), bs)
                sparse_avg.update(acc_rx.item(), bs)

                # 对三种图像进行分类
                out_res = model.classifier(x - gx)  # 残差图像分类
                out_orig = model.classifier(x)      # 原始图像分类
                out_recon = model.classifier(gx)    # 重建图像分类

                # 计算三种分类准确率
                prec1_res, _, _, _ = accuracy(out_res.data, y.data, topk=(1, 5))
                prec1_orig, _, _, _ = accuracy(out_orig.data, y.data, topk=(1, 5))
                prec1_recon, _, _, _ = accuracy(out_recon.data, y.data, topk=(1, 5))

                top1_res.update(prec1_res.item(), bs)
                top1_orig.update(prec1_orig.item(), bs)
                top1_recon.update(prec1_recon.item(), bs)

                if hi is not None and mu is not None and logvar is not None:
                    try:
                        tc = total_correlation(hi, mu, logvar) / bs / args.dim
                        TC.update(tc.item(), bs)
                    except Exception as e:
                        pass
                else:
                    pass

                pbar.set_postfix(
                    RecAcc=acc_avg.avg, 
                    ResClsAcc=top1_res.avg,
                    OrigClsAcc=top1_orig.avg,
                    ReconClsAcc=top1_recon.avg,
                    TC=TC.avg if TC.count > 0 else float('nan')
                )

            except Exception as e:
                print(f"\nError during testing batch {batch_idx}: {e}")
                continue

    # 记录所有指标到WandB
    log_data = {
        'test_rec_acc_avg': acc_avg.avg,
        'test_sparse_avg': sparse_avg.avg,
        'test_residual_class_acc': top1_res.avg,
        'test_original_class_acc': top1_orig.avg,
        'test_reconstructed_class_acc': top1_recon.avg,
    }
    if TC.count > 0:
        log_data['test_TC'] = TC.avg

    # 调用图像记录
    log_individual_images(
        dataloader=testloader,
        model=model,
        args=args,
        batch_num=2,
        save_local=True
    )

    # 提交最终日志
    wandb.log(log_data, commit=True)

    # 打印最终结果摘要
    print(f"\n| Test Results\tRec Acc: {acc_avg.avg:.4f}")
    print(f"| Classification Results:")
    print(f"|\tResidual Acc: {top1_res.avg:.4f}")
    print(f"|\tOriginal Acc: {top1_orig.avg:.4f}")
    print(f"|\tReconstructed Acc: {top1_recon.avg:.4f}")
    print(f"|\tTC: {TC.avg if TC.count > 0 else 'N/A'}")


# --- Main Function (Modified for Testing Only, Epoch Extraction Removed) ---
def main(args):
    CNN_embed_dim = args.dim
    feature_dim = args.fdim
    setup_logger(args.save_dir)
    use_cuda = torch.cuda.is_available()

    print('\n[Phase 1] : Data Preparation')
    test_path = './data/%s/test' % (args.dataset)
    if args.dataset == 'mstar':
        num_class = 10
        mean = (0.13620707,)
        std = (0.11149936,)
    elif args.dataset == 'FUSAR':
        num_class = 10
        mean = (0.10994489,)
        std = (0.10608266,)
    # ... (other dataset elif blocks) ...
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"| Preparing {args.dataset} dataset...")
    train_path = './data/%s/train' % (args.dataset)
    test_path = './data/%s/test' % (args.dataset)

    if args.dataset == 'mstar':
        num_class = 10
        mean = (0.13620707)
        std = (0.11149936)
        print("| Preparing mstar dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'acd':
        num_class = 6
        mean = (0.0565084)
        std = (0.09828296)
        print("| Preparing acd dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'opensar':
        num_class = 6
        mean = (0.05527896)
        std = (0.06979666)
        print("| Preparing opensar dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'FUSAR':
        num_class = 10
        mean = (0.10994489,)
        std = (0.10608266,)
        print("| Preparing FUSAR dataset...")
        sys.stdout.write("| ")

    _, test_loader = get_dataloader(args.dataset, args.batch_size, args.size,
                                    train_path=train_path, test_path=test_path,
                                    mean=mean, std=std)

    print('\n[Phase 2] : Model setup and Loading Weights')
    # 根据backbone参数选择模型架构
    if args.backbone == 'resnet':
        model = CVAE_SAR_Res(d=feature_dim, z=CNN_embed_dim, num_classes=num_class)
    elif args.backbone == 'wid':
        model = CVAE_SAR_Wid(d=feature_dim, z=CNN_embed_dim, num_classes=num_class)
    elif args.backbone == 'vgg':
        model = CVAE_SAR_vgg(d=feature_dim, z=CNN_embed_dim, num_classes=num_class)
    # model = CVAE_SAR_Res(d=feature_dim, z=CNN_embed_dim, num_classes=num_class)
    print(f"Using model architecture: {type(model).__name__}")

    if not os.path.exists(args.model_path):
        print(f"Error: Model weights file not found at {args.model_path}")
        return

    print(f"Loading pre-trained model from: {args.model_path}")
    try:
        map_location = 'cuda' if use_cuda else 'cpu'
        state_dict = torch.load(args.model_path, map_location=map_location)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_data_parallel = any(k.startswith('module.') for k in state_dict.keys())
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        if is_data_parallel:
            print("Detected DataParallel prefix ('module.') in saved state dict keys.")
        model.load_state_dict(new_state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
         print(f"Error loading model weights: {e}. Exiting.")
         return

    if use_cuda:
        print("Using CUDA")
        model.cuda()
        if is_data_parallel and torch.cuda.device_count() > 1:
             print(f"Using {torch.cuda.device_count()} GPUs for testing via DataParallel!")
             model = torch.nn.DataParallel(model)
        elif torch.cuda.device_count() > 1:
            print(f"Multiple GPUs ({torch.cuda.device_count()}) detected, but DP wrap not automatically applied.")
        cudnn.benchmark = True

    print('\n[Phase 3] : Testing')
    # Removed epoch extraction logic
    # Call test function (no longer passes epoch)
    test(model, test_loader, args)

    print("\nTesting finished.")
    wandb.finish()


# --- Argument Parser and Main Execution (Unchanged from previous version) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SAR CVAE Testing Script')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained model file (.pth)')
    parser.add_argument('--save_dir', default='./results_SAR/test_run/', type=str, help='Directory to save testing results (like images)')
    parser.add_argument('--dataset', default='mstar', type=str, help='dataset = [mstar/acd/opensar/FUSAR]')
    parser.add_argument('--size', default=128, type=int, help='Image size used during training/testing')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for testing')
    parser.add_argument('--dim', default=2048, type=int, help='VAE latent dimension (z) used during training')
    parser.add_argument('--fdim', default=32, type=int, help='Feature dimension (d) used during training')
    parser.add_argument('--seed', default=666, type=int, help='Random seed for reproducibility')
    parser.add_argument('--backbone', default='resnet', type=str, choices=['vae', 'wid', 'resnet', 'vgg', 'preres', 'mobile', 'densenet'], 
                        help='选择模型的backbone架构: vae=CVAE_SAR, wid=CVAE_SAR_Wid, resnet=CVAE_SAR_Res, vgg=CVAE_SAR_vgg, preres=CVAE_SAR_PreActResNet, mobile=CVAE_SAR_mobile, densenet=CVAE_SAR_densenet')
    args = parser.parse_args()

    model_filename_base = os.path.splitext(os.path.basename(args.model_path))[0]
    wandb_run_name = f"test_{args.dataset}_{model_filename_base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(config=args, project="SAR_CVAE_Testing_Runs", name=wandb_run_name, mode="offline")

    set_random_seed(args.seed) # Assuming set_random_seed is defined

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created save directory: {args.save_dir}")

    main(args)