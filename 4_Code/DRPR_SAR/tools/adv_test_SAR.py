from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms

import os
# 移除 wandb API key 设置
import time
import argparse
import datetime
from torch.autograd import Variable
import pdb
import sys
# 移除 wandb 导入

sys.path.append('.')
from typing import Dict, List
from networks_V1.adv_vae_SAR import *
from perceptual_advex.attacks import *
# from advex.attacks import *
# from utils.normalize import *
from utils.set import *
from utils.randaugment4fixmatch import RandAugmentMC
from utils.SAR_loader import get_dataloader
# from perceptual_advex.attacks import StAdvAttack

# normalize = SARNORMALIZE(128)

import matplotlib.pyplot as plt
import numpy as np

# Helper function to save single-channel images
def save_image(tensor, filename):
    """Saves a single-channel image tensor (C, H, W) or (H, W) to a file."""
    # Ensure tensor is on CPU, detached, and converted to numpy
    image = tensor.detach().cpu().numpy()
    # Squeeze singleton dimensions (like channel dimension for grayscale)
    if image.ndim == 3: # If shape is (1, H, W)
        image = image.squeeze(0) # Remove channel dim -> (H, W)
    # Ensure it's float32 for imsave, handle potential range issues if needed
    image = image.astype(np.float32)
    # Use imsave - it handles grayscale and colormapping automatically
    # cmap='gray' ensures it's saved as grayscale
    # vmin/vmax can be used to normalize if needed, but None often works well.
    plt.imsave(filename, image, cmap='gray', vmin=None, vmax=None)
    # Optional: print message
    # print(f"Saved image: {filename}")

class VAEClassifier(nn.Module):
    def __init__(self, vae, classifier):
        super().__init__()
        self.vae = vae
        self.classifier = classifier
    def forward(self, x):
        gx, _, _ = self.vae(x)
        return self.classifier(gx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch SAR Testing')
    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')
    parser.add_argument('--dim', default=2048, type=int, help='CNN_embed_dim')
    parser.add_argument('--fdim', default=32, type=int, help='featdim')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')


    #FUSAR参数
    parser.add_argument("--model_path", type=str, default="./results/mstar/Dill_mstar_resnet50_0.2_0.2_0.2_0508V2/robust_model_g_epoch301.pth")
    parser.add_argument("--vae_path", type=str, default="./results/mstar/Dill_mstar_resnet50_0.2_0.2_0.2_0508V2/robust_vae_epoch301.pth")

    parser.add_argument('--dataset', default='mstar', type=str, help='dataset = [mstar/acd/opensar/FUSAR]')
    parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
    parser.add_argument('--size', default=128, type=int)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    print("| Preparing SAR dataset...")
    sys.stdout.write("| ")

    train_path = './data/%s/train' % (args.dataset)
    test_path = './data/%s/test' % (args.dataset)

    # 根据数据集选择均值和标准差
    if args.dataset == 'mstar':
        num_class = 10
        mean = (0.13620707,)
        std = (0.11149936,)
        print("| Preparing mstar dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'acd':
        num_class = 6
        mean = (0.0565084,)
        std = (0.09828296,)
        print("| Preparing acd dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'opensar':
        num_class = 6
        mean = (0.05527896,)
        std = (0.06979666,)
        print("| Preparing opensar dataset...")
        sys.stdout.write("| ")
    elif args.dataset == 'FUSAR':
        num_class = 10
        mean = (0.10994489,)
        std = (0.10608266,)
        print("| Preparing FUSAR dataset...")
        sys.stdout.write("| ")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # normalize = SARNORMALIZE(128, args.dataset)

    train_loader, test_loader = get_dataloader(args.dataset, args.batch_size, args.size, train_path=train_path,
                                               test_path=test_path, mean=mean, std=std)

    # cd_vae = CD_VAE_SAR_Res(args.vae_path, args.model_path, args.num_classes, args.dataset)#networks_V1.adv_vae_SAR. CD_VAE_SAR_Res
    # cd_vae = CD_VAE_SAR_Wid(args.vae_path, args.model_path, args.num_classes,
    #                         args.dataset)  # networks_V1.adv_vae_SAR. CD_VAE_SAR_Res

    # vae = CVAE_SAR_Res(d=2048, z=32, with_classifier=False)# networks_V1.adv_vae_SAR. CVAE_SAR_Res
    
    # cd_vae = CD_VAE_SAR_Res(args.vae_path, args.model_path, args.num_classes, args.dataset)#
    cd_vae = CD_VAE_SAR_densenet(args.vae_path, args.model_path, args.num_classes, args.dataset)
    
    # cd_vae = CD_VAE_SAR_preres(args.vae_path, args.model_path, args.num_classes, args.dataset)  #

    # 移除 wandb.init(config=args)
    if use_cuda:
        cd_vae.cuda()
        cudnn.benchmark = True

    cd_vae.eval()

    attack_names: List[str] = args.attacks
    batches_correct: Dict[str, List[torch.Tensor]] = {attack_name: [] for attack_name in attack_names}

    # ========== 关键修改：用 get_attack 初始化攻击对象 ==========
    attacks = []
    for attack_name in attack_names:
        # 用VAE净化+分类器的组合模型
        vae_classifier = VAEClassifier(cd_vae.vae, cd_vae.model)
        attack = get_attack(attack_name, vae_classifier, num_class=num_class)
        attacks.append(attack)

    # ========== 遍历测试数据并进行攻击 ==========
    # Comment out image saving directory creation
    save_img_dir = f'./saved_images_{args.dataset}_{"_".join(attack_names)}'
    os.makedirs(save_img_dir, exist_ok=True)
    print(f"Saving images to: {save_img_dir}")

    for batch_index, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating Batches")):
        # Comment out image saving flag
        SAVE_IMAGES = batch_index < 2 # Example: Save images only for the first 2 batches

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # --- Prepare normalized clean inputs for later use ---
        # with torch.no_grad():
        #     normalized_inputs = normalize(inputs)

        # 对每个攻击生成对抗样本并评估
        # 在攻击循环中添加特殊处理
        for attack_name, attack in zip(attack_names, attacks):
            if attack_name == 'sparseRS':
                # sparseRS 需要特殊处理，因为它在CPU上运行
                inputs_cpu = inputs.cpu()
                labels_cpu = labels.cpu()
                
                # 创建CPU版本的模型包装器
                def cpu_model_wrapper(x):
                    with torch.no_grad():
                        x_gpu = x.cuda() if torch.cuda.is_available() else x
                        gx, _, _ = cd_vae.vae(x_gpu)
                        logits = cd_vae.model(gx)
                        return logits.cpu()
                
                # 重新创建sparseRS攻击，使用CPU模型包装器
                sparse_attack = RSAttack(
                    predict=cpu_model_wrapper,
                    norm='L0',
                    n_queries=5000,  # 可以调整查询次数
                    eps=150,         # 可以调整扰动像素数
                    p_init=0.3,
                    n_restarts=1,
                    seed=0,
                    verbose=False,
                    targeted=False,
                    loss='margin',
                    device='cpu'
                )
                
                # 执行攻击
                _, adv_inputs_cpu = sparse_attack.perturb(inputs_cpu, labels_cpu)
                adv_inputs = adv_inputs_cpu.cuda() if torch.cuda.is_available() else adv_inputs_cpu
            else:
                # 其他攻击的正常处理
                adv_inputs = attack(inputs, labels)
            
            # 后续处理保持不变
            with torch.no_grad():
                gx, _, _ = cd_vae.vae(adv_inputs)
                logits = cd_vae.model(gx)
                rx = adv_inputs - gx
        
            # --- Calculate delta ---
            delta = adv_inputs - inputs

            # --- Comment out the image saving block ---
            if SAVE_IMAGES:
                # Get the first sample from the batch
                idx_to_save = 0
                input_sample = inputs[idx_to_save]
                adv_input_sample = adv_inputs[idx_to_save]
                delta_sample = delta[idx_to_save]
                gx_sample = gx[idx_to_save]
                rx_sample = rx[idx_to_save]
            
                # Define filenames
                base_filename = f"{save_img_dir}/{attack_name}_batch{batch_index}_sample{idx_to_save}"
                filename_orig = f"{base_filename}_orig.png"
                filename_adv = f"{base_filename}_adv.png"
                filename_delta = f"{base_filename}_delta.png"
                filename_gx = f"{base_filename}_gx.png"
                filename_rx = f"{base_filename}_rx_normspace.png"
            
                # Savade images using the helper function
                save_image(input_sample, filename_orig)
                save_image(adv_input_sample, filename_adv)
                save_image(delta_sample, filename_delta)
                save_image(gx_sample, filename_gx)
                save_image(rx_sample, filename_rx)
            # --- End of commented out block ---


            # --- Evaluate accuracy (remains the same) ---
            batch_correct = (logits.argmax(1) == labels).detach()

            # 记录结果 (remains the same)
            batches_correct[attack_name].append(batch_correct)


    # ========== 汇总统计结果 (remains the same) ==========
    print('OVERALL')
    accuracies = []
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              sep='\t')
        accuracies.append(accuracy)


