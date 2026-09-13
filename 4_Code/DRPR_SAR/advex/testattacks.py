import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For default step size calculation if needed
from tqdm import TqdmWarning # To potentially ignore tqdm ExperimentalWarning
import warnings
import torchattacks
# 假设你的 manual_attacks.py 在 advex 目录下
from advex.manual_attacks import get_manual_attack 

warnings.filterwarnings("ignore", category=TqdmWarning) # Optional: Ignore tqdm warnings if they appear


class ManualAttackWrapper:
    """
    Wraps manual implementations of common adversarial attacks, 
    adapted for normalized SAR data using min/max value clipping.
    """
    def __init__(self, 
                 model, 
                 criterion, 
                 eps=8/255, 
                 alpha_linf=2/255, 
                 steps_linf=10, 
                 eps_l2=1.0, 
                 alpha_l2=0.1, 
                 steps_l2=10, 
                 cw_c=1.0, 
                 cw_steps=100, 
                 cw_lr=0.01, 
                 valid_min_val=-1.0, 
                 valid_max_val=1.0,
                 num_classes=10): # Added num_classes for CW targeted default
        """
        Args:
            model: The target PyTorch model (should be in eval mode before attack).
            criterion: The loss function (e.g., nn.CrossEntropyLoss).
            eps: Epsilon for L-infinity attacks.
            alpha_linf: Step size for L-infinity PGD/BIM.
            steps_linf: Number of steps for L-infinity PGD/BIM.
            eps_l2: Epsilon for L2 attacks.
            alpha_l2: Step size for PGD L2.
            steps_l2: Number of steps for PGD L2.
            cw_c: C constant for CW attack.
            cw_steps: Number of steps for CW attack.
            cw_lr: Learning rate for CW attack optimizer.
            valid_min_val: Minimum valid value in the normalized space.
            valid_max_val: Maximum valid value in the normalized space.
            num_classes: Number of classes in the dataset (for CW targeted).
        """
        self.model = model
        self.criterion = criterion
        self.eps = eps
        self.alpha_linf = alpha_linf
        self.steps_linf = steps_linf
        self.eps_l2 = eps_l2
        self.alpha_l2 = alpha_l2
        self.steps_l2 = steps_l2
        self.cw_c = cw_c
        self.cw_steps = cw_steps
        self.cw_lr = cw_lr
        self.valid_min_val = valid_min_val
        self.valid_max_val = valid_max_val
        self.num_classes = num_classes # Store num_classes


    def _project_linf(self, delta):
        """Helper function to project perturbation into L-inf ball"""
        return torch.clamp(delta, -self.eps, self.eps)

    def _project_l2(self, delta):
        """Helper function to project perturbation onto L2 ball"""
        batch_size = delta.shape[0]
        # Calculate L2 norm for each sample in the batch
        norms = torch.linalg.norm(delta.view(batch_size, -1), dim=1, ord=2)
        # Avoid division by zero
        norms = torch.max(norms, torch.tensor(1e-12, device=delta.device)) 
        # Calculate scaling factor, ensuring it doesn't exceed 1
        factor = torch.minimum(self.eps_l2 / norms, torch.ones_like(norms))
        # Apply scaling factor
        delta *= factor.view(batch_size, 1, 1, 1)
        return delta

    def _clip_to_valid_range(self, images):
         """Clips images to the valid normalized range"""
         return torch.clamp(images, min=self.valid_min_val, max=self.valid_max_val)

    # --- FGSM Implementation ---
    def _attack_fgsm(self, images, labels):
        # Ensure model is in eval mode (important if BN/Dropout exist)
        # self.model.eval() # Consider setting eval mode outside the attack function call
        
        images_clone = images.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(images_clone)
        loss = self.criterion(outputs, labels)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward() # Assuming criterion output is scalar or summed before backward if not

        if images_clone.grad is None:
            print("!!!! FGSM WARNING: Gradient is None. Returning original images. !!!!")
            return images.clone().detach()

        grad = images_clone.grad.detach()
        
        # FGSM perturbation step
        delta_raw = self.eps * grad.sign()
        adv_images_unclipped = images_clone + delta_raw
        
        # Clip to valid range
        adv_images = self._clip_to_valid_range(adv_images_unclipped)
        
        return adv_images.detach()

    # --- PGD L-infinity Implementation ---
    def _attack_pgd_linf(self, images, labels):
        # self.model.eval()
        images = images.detach() # Make sure original images don't require gradients

        # Random Start (essential for PGD)
        delta_init = torch.empty_like(images).uniform_(-self.eps, self.eps)
        # Clip the initial adversarial example
        adv_images = self._clip_to_valid_range(images + delta_init)
        # Project the initial delta back into the L-inf ball
        delta = self._project_linf(adv_images - images) 
        
        delta = delta.detach().requires_grad_(True)

        for _ in range(self.steps_linf):
            adv_images_current = images + delta # Use current delta
            
            # Forward pass on current adversary
            outputs = self.model(adv_images_current)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            # Aggregate loss before backward if criterion returns per-sample loss
            if loss.dim() > 0: 
                loss.sum().backward() 
            else:
                 loss.backward()

            if delta.grad is None:
                 print("!!!! PGD-Linf WARNING: Gradient is None. Stopping attack early. !!!!")
                 break 
            
            grad = delta.grad.detach()
            
            # PGD Step
            step_delta = self.alpha_linf * grad.sign()
            delta.data = delta.data + step_delta      # Take the step
            delta.data = self._project_linf(delta.data) # Project delta back

            # Clip the *resulting image* and update delta accordingly
            adv_images_clipped = self._clip_to_valid_range(images + delta.data)
            delta.data = adv_images_clipped - images
            
            # Zero gradient for next iteration
            delta.grad.zero_()

        final_adv_images = self._clip_to_valid_range(images + delta.detach())
        return final_adv_images

    # --- BIM Implementation (PGD L-inf without Random Start) ---
    def _attack_bim(self, images, labels):
        # self.model.eval()
        images = images.detach()
        
        # Start from zero perturbation
        delta = torch.zeros_like(images, requires_grad=True)

        for _ in range(self.steps_linf):
            adv_images_current = images + delta
            
            outputs = self.model(adv_images_current)
            loss = self.criterion(outputs, labels)
            
            self.model.zero_grad()
            if loss.dim() > 0: loss.sum().backward()
            else: loss.backward()

            if delta.grad is None:
                 print("!!!! BIM WARNING: Gradient is None. Stopping attack early. !!!!")
                 break
                 
            grad = delta.grad.detach()
            
            # BIM Step
            step_delta = self.alpha_linf * grad.sign()
            delta.data = delta.data + step_delta
            delta.data = self._project_linf(delta.data) # Project delta back

            # Clip the resulting image and update delta
            adv_images_clipped = self._clip_to_valid_range(images + delta.data)
            delta.data = adv_images_clipped - images
            
            delta.grad.zero_()

        final_adv_images = self._clip_to_valid_range(images + delta.detach())
        return final_adv_images

    # --- PGD L2 Implementation ---
    def _attack_pgd_l2(self, images, labels):
        # self.model.eval()
        images = images.detach()
        batch_size = images.shape[0]

        # Optional: Random Start for PGD-L2
        # delta_init = torch.randn_like(images) 
        # delta_init = self._project_l2(delta_init) # Start on the surface
        # adv_images = self._clip_to_valid_range(images + delta_init)
        # delta = adv_images - images
        # delta = self._project_l2(delta) # Ensure initial delta obeys constraint
        
        # Start from zero perturbation for simplicity here
        delta = torch.zeros_like(images, requires_grad=True)

        for _ in range(self.steps_l2):
            adv_images_current = images + delta
            
            outputs = self.model(adv_images_current)
            loss = self.criterion(outputs, labels)
            
            self.model.zero_grad()
            if loss.dim() > 0: loss.sum().backward()
            else: loss.backward()
            
            if delta.grad is None:
                 print("!!!! PGD-L2 WARNING: Gradient is None. Stopping attack early. !!!!")
                 break

            grad = delta.grad.detach()
            
            # PGD L2 Step - Normalize gradient step by L2 norm
            grad_norms = torch.linalg.norm(grad.view(batch_size, -1), dim=1, ord=2) + 1e-12
            step_delta = self.alpha_l2 * grad / grad_norms.view(-1, 1, 1, 1)

            delta.data = delta.data + step_delta      # Take the step
            delta.data = self._project_l2(delta.data) # Project delta back onto L2 ball

            # Clip the resulting image and update delta
            adv_images_clipped = self._clip_to_valid_range(images + delta.data)
            delta.data = adv_images_clipped - images

            delta.grad.zero_()

        final_adv_images = self._clip_to_valid_range(images + delta.detach())
        return final_adv_images

    # --- CW L2 Implementation ---
    def _attack_cw_l2(self, images, labels, targeted=False):
        # self.model.eval() # Ensure model is in eval
        device = images.device
        batch_size = images.size(0)
        min_val, max_val = self.valid_min_val, self.valid_max_val

        # Inverse tanh transform helper
        def arctanh(x):
            # Map from [min_val, max_val] to ~(-1, 1)
            x_mapped = (x.clamp(min_val, max_val) - min_val) / (max_val - min_val) * 2 - 1
            return 0.5 * torch.log((1 + x_mapped.clamp(-1+1e-6, 1-1e-6)) / (1 - x_mapped.clamp(-1+1e-6, 1-1e-6)))

        # Initial w based on images
        w = arctanh(images).detach().requires_grad_(True)
        
        # Setup optimizer for w
        optimizer = torch.optim.Adam([w], lr=self.cw_lr)

        # Determine target labels if needed
        target_labels = labels # Default: untargeted
        if targeted:
            # Simple targeted: next class modulo num_classes
            target_labels = (labels + 1) % self.num_classes 

        # Margin loss helper (using CW's f_6 definition)
        def margin_loss(outputs, current_labels, is_targeted):
             one_hot_labels = torch.nn.functional.one_hot(current_labels, num_classes=self.num_classes).float()
             correct_logits = (one_hot_labels * outputs).sum(dim=1)
             # Find the highest logit for incorrect classes
             other_logits_masked = outputs - one_hot_labels * 1e4 # Mask out correct class
             max_other_logits = other_logits_masked.max(dim=1)[0]
             
             if is_targeted:
                 # Want target_logit > max_other_logit
                 return torch.clamp(max_other_logits - correct_logits, min=0) 
             else: # Untargeted
                 # Want max_other_logit > correct_logit
                 return torch.clamp(correct_logits - max_other_logits, min=0)


        for step in range(self.cw_steps):
            # Map w back to image space
            adv_images_tanh = torch.tanh(w)
            adv_images = (adv_images_tanh + 1) / 2 * (max_val - min_val) + min_val
            # Clip to ensure validity before passing to model (important!)
            adv_images = self._clip_to_valid_range(adv_images) 
            
            # Calculate L2 distortion
            l2_distortion = ((adv_images - images) ** 2).view(batch_size, -1).sum(dim=1)

            # Calculate classification loss (margin loss)
            outputs = self.model(adv_images) # Pass clipped adv_images
            
            # Use target_labels if targeted, else original labels
            loss_labels = target_labels if targeted else labels
            c_loss = margin_loss(outputs, loss_labels, targeted)

            # Total CW loss
            total_loss_per_sample = l2_distortion + self.cw_c * c_loss
            total_loss_batch = total_loss_per_sample.sum() # Aggregate for backward

            # Optimization step
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            # Optional: logging Loss components every N steps
            if step % (self.cw_steps // 5) == 0 or step == self.cw_steps - 1:
                 print(f"    CW_L2 Step {step}: mean L2_dist={l2_distortion.mean().item():.4f}, mean C_loss={c_loss.mean().item():.4f}")


        # Final adversarial image after optimization
        final_adv_images_tanh = torch.tanh(w).detach()
        final_adv_images = (final_adv_images_tanh + 1) / 2 * (max_val - min_val) + min_val
        final_adv_images = self._clip_to_valid_range(final_adv_images) # Final clip

        print("    DEBUG CW_L2: Exiting _attack_cw_l2 function.")
        return final_adv_images.detach()

# --- Factory Function ---
def get_manual_attack(atk_name, 
                      model, 
                      criterion, 
                      eps=8/255, 
                      alpha_linf=2/255, 
                      steps_linf=10, 
                      eps_l2=1.0, 
                      alpha_l2=0.1, 
                      steps_l2=10, 
                      cw_c=1.0, 
                      cw_steps=100, 
                      cw_lr=0.01, 
                      valid_min_val=-1.0, 
                      valid_max_val=1.0,
                      num_classes=10):
    """
    Factory function to get a manual attack method handle.

    Args:
        atk_name (str): Name of the attack ('FGSM', 'PGD', 'BIM', 'PGD_L2', 'CW', 'NoAttack'). Case-insensitive.
        model: Target model.
        criterion: Loss function.
        ... other attack parameters ...
        valid_min_val: Min value for clipping in normalized space.
        valid_max_val: Max value for clipping in normalized space.
        num_classes: Number of classes (needed for CW targeted default).

    Returns:
        callable: A function that takes (images, labels) and returns adversarial images.
    """
    wrapper = ManualAttackWrapper(
        model=model, criterion=criterion, eps=eps, alpha_linf=alpha_linf, steps_linf=steps_linf,
        eps_l2=eps_l2, alpha_l2=alpha_l2, steps_l2=steps_l2, cw_c=cw_c, cw_steps=cw_steps, cw_lr=cw_lr,
        valid_min_val=valid_min_val, valid_max_val=valid_max_val, num_classes=num_classes
    )
    
    atk_name_upper = atk_name.upper()

    if atk_name_upper == 'FGSM':
        print(f"| Using Manual Attack: FGSM (eps={eps:.4f})")
        return wrapper._attack_fgsm
    elif atk_name_upper == 'PGD':
        print(f"| Using Manual Attack: PGD L-inf (eps={eps:.4f}, steps={steps_linf}, alpha={alpha_linf:.4f})")
        return wrapper._attack_pgd_linf
    elif atk_name_upper == 'BIM':
        print(f"| Using Manual Attack: BIM (PGD L-inf w/o rand_start) (eps={eps:.4f}, steps={steps_linf}, alpha={alpha_linf:.4f})")
        return wrapper._attack_bim
    elif atk_name_upper == 'PGD_L2':
        print(f"| Using Manual Attack: PGD L2 (eps={eps_l2:.4f}, steps={steps_l2}, alpha={alpha_l2:.4f})")
        return wrapper._attack_pgd_l2
    elif atk_name_upper == 'CW': # Assuming CW means CW-L2
        print(f"| Using Manual Attack: CW L2 (c={cw_c}, steps={cw_steps}, lr={cw_lr})")
        # If you need to pass targeted=True for CW, you'd need another argument
        # or a separate attack name like 'CW_TARGETED'
        return wrapper._attack_cw_l2 
    elif atk_name_upper == 'NOATTACK':
        print("| Using Manual Attack: NoAttack")
        return lambda images, labels: images.clone().detach()
    else:
        raise ValueError(f"Unknown manual attack name: {atk_name}. Choose from ['FGSM', 'PGD', 'BIM', 'PGD_L2', 'CW', 'NoAttack']")

# 需要传入更多参数给 get_manual_attack
def get_combined_attack(atk, model, criterion, # for manual
                        num_class, # for manual cw and original torchattacks jsma/autoattack
                        eps=8/255, alpha_linf=2/255, steps_linf=10, # for manual linf
                        eps_l2=1.0, alpha_l2=0.1, steps_l2=10,     # for manual l2
                        cw_c=1.0, cw_steps=100, cw_lr=0.01,        # for manual cw
                        valid_min_val=-1.0, valid_max_val=1.0      # for manual clipping
                        ):
    
    manual_attack_names = ['FGSM', 'PGD', 'BIM', 'PGD_L2', 'CW', 'NOATTACK']
    torchattacks_auto = ['AUTOLINFATTACK', 'AUTOL2ATTACK'] # torchattacks AutoAttack names
    # 其他 torchattacks 库的攻击名称...

    atk_upper = atk.upper() # 转为大写以便比较

    if atk_upper in [name.upper() for name in manual_attack_names]:
        print(f"Using Manual Attack Implementation for: {atk}")
        attack_fn = get_manual_attack(
            atk_name=atk, model=model, criterion=criterion, 
            eps=eps, alpha_linf=alpha_linf, steps_linf=steps_linf,
            eps_l2=eps_l2, alpha_l2=alpha_l2, steps_l2=steps_l2,
            cw_c=cw_c, cw_steps=cw_steps, cw_lr=cw_lr,
            valid_min_val=valid_min_val, valid_max_val=valid_max_val,
            num_classes=num_class
        )
        # 注意：get_manual_attack 返回的是一个函数，可以直接调用
        return attack_fn 
        
    elif atk_upper == 'AUTOLINFATTACK':
        print(f"Using torchattacks.AutoAttack: Linf")
        # 使用 torchattacks.AutoAttack，需要传入 model, norm, eps, n_classes
        # 注意：torchattacks 的 AutoAttack 可能需要模型返回 logits
        # 可能需要调整 torchattacks 的版本或参数以适应你的模型和数据
        attack_obj = torchattacks.AutoAttack(model, norm='Linf', eps=eps, n_classes=num_class, version='standard') # 使用标准版本
        # 返回 torchattacks 对象，调用时使用 attack_obj(images, labels)
        return attack_obj
        
    elif atk_upper == 'AUTOL2ATTACK':
        print(f"Using torchattacks.AutoAttack: L2")
        # L2 的 eps 通常不同，从你的原始代码看是 0.05？或者需要一个单独参数
        auto_eps_l2 = 0.5 # 示例值，需要确认
        attack_obj = torchattacks.AutoAttack(model, norm='L2', eps=auto_eps_l2, n_classes=num_class, version='standard') 
        return attack_obj

    # --- 处理原始代码中其他 torchattacks 攻击 ---
    elif atk_upper == 'JSMA':
         print(f"Using torchattacks.JSMA")
         attack_obj = torchattacks.JSMA(model, gamma=0.01, num_classes=num_class) 
         return attack_obj
    elif atk_upper == 'DEEPFOOL':
         print(f"Using torchattacks.DeepFool")
         attack_obj = torchattacks.DeepFool(model, steps=20, overshoot=0.08)
         return attack_obj
    # ... 添加其他 torchattacks 攻击的处理 ...
    
    else:
        raise ValueError(f"Attack '{atk}' not recognized by get_combined_attack")

# --- Example Usage (inside your main script) ---
if __name__ == '__main__':
    # --- Dummy Setup (Replace with your actual setup) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Define your Target Model (e.g., ResNet50 adapted for 1 channel)
    # Make sure the model is adapted for single channel input if needed
    class DummyModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, 1, 1) # Example: 1 input channel
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(16, num_classes)
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)
            
    model = DummyModel(num_classes=10).to(device).eval() # <<-- IMPORTANT: Set to eval() mode 
    
    # 2. Define Loss Criterion
    criterion = nn.CrossEntropyLoss()
    
    # 3. Define Normalization Range (Example for MSTAR)
    mean_val, std_val = 0.1362, 0.1115
    valid_min_val = (0.0 - mean_val) / std_val
    valid_max_val = (1.0 - mean_val) / std_val
    num_classes=10

    # 4. Get an attack function handle using the factory
    pgd_attack_fn = get_combined_attack(
        atk='PGD', 
        model=model, 
        criterion=criterion,
        num_class=num_classes,
        eps=8/255, 
        steps_linf=20, 
        alpha_linf=(8/255)/4, # Example alpha
        valid_min_val=valid_min_val,
        valid_max_val=valid_max_val
    )
    
    cw_attack_fn = get_combined_attack(
        atk='CW',
        model=model,
        criterion=criterion, # CW implementation uses criterion implicitly via margin loss logic
        num_class=num_classes,
        cw_c=1.0,
        cw_steps=50,
        cw_lr=0.02,
        valid_min_val=valid_min_val,
        valid_max_val=valid_max_val
    )

    # 5. Create Dummy Data (Replace with your actual data loader)
    dummy_images = torch.randn(4, 1, 128, 128, device=device) # Example Batch [B, C, H, W]
    # Apply your normalization to dummy images if needed for testing
    # dummy_images_normalized = (dummy_images - mean_val) / std_val 
    dummy_images_normalized = dummy_images # Assume normalized for this example
    dummy_labels = torch.randint(0, 10, (4,), device=device)

    # 6. Run the attack
    print("\nRunning PGD Attack...")
    adversarial_images_pgd = pgd_attack_fn(dummy_images_normalized, dummy_labels)
    print("PGD Attack Done. Shape:", adversarial_images_pgd.shape)
    # Verify perturbation norm
    delta_pgd = adversarial_images_pgd - dummy_images_normalized
    print(f"PGD Delta Linf: {delta_pgd.abs().max().item():.6f}")


    print("\nRunning CW Attack...")
    adversarial_images_cw = cw_attack_fn(dummy_images_normalized, dummy_labels)
    print("CW Attack Done. Shape:", adversarial_images_cw.shape)
    # Verify perturbation norm
    delta_cw = adversarial_images_cw - dummy_images_normalized
    print(f"CW Delta L2 (avg per image): {torch.linalg.norm(delta_cw.view(4, -1), dim=1).mean().item():.4f}")
