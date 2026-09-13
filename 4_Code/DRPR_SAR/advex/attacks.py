import sys
import os
import torch
import functools
from torch import nn
from operator import mul
from torch import optim

# Optional sparse-rs integration. Set SPARSE_RS_ROOT when using it.
sparse_rs_root = os.environ.get("SPARSE_RS_ROOT")
if sparse_rs_root:
    sys.path.append(sparse_rs_root)
    # from rs_attacks import RSAttack
    sys.path.remove(sparse_rs_root)

# from advex_uar.common.pyt_common import get_attack as get_uar_attack
# from advex_uar.attacks.attacks import InverseImagenetTransform

# from .perceptual_attacks import *
# from .utilities import LambdaLayer
# from . import utilities

# mister_ed
# from recoloradv.mister_ed import loss_functions as lf
# from recoloradv.mister_ed import adversarial_training as advtrain
# from recoloradv.mister_ed import adversarial_perturbations as ap 
# from recoloradv.mister_ed import adversarial_attacks as aa
# from recoloradv.mister_ed import spatial_transformers as st

# # ReColorAdv
# from recoloradv import perturbations as pt
# from recoloradv import color_transformers as ct
# from recoloradv import color_spaces as cs


import torchattacks

PGD_ITERS = 20
# DATASET_NUM_CLASSES = {
#     'cifar': 10,
#     'imagenet100': 100,
#     'imagenet': 1000,
#     'bird_or_bicycle': 2,
# }

DATASET_NUM_CLASSES = {
    'mstar': 10,
    'acd': 6,
    'opensar': 6,
    'FUSAR':10
}


class NoAttack(nn.Module):
    """
    Attack that does nothing.
    """

    def __init__(self, model=None):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels):
        return inputs


class MisterEdAttack(nn.Module):
    """
    Base class for attacks using the mister_ed library.
    """

    def __init__(self, model, threat_model, randomize=False,
                 perturbation_norm_loss=False, lr=0.001, random_targets=False,
                 num_classes=None, **kwargs):
        super().__init__()

        self.model = model
        self.normalizer = nn.Identity()

        self.threat_model = threat_model
        self.randomize = randomize
        self.perturbation_norm_loss = perturbation_norm_loss
        self.attack_kwargs = kwargs
        self.lr = lr
        self.random_targets = random_targets
        self.num_classes = num_classes

        self.attack = None

    def _setup_attack(self):
        cw_loss = lf.CWLossF6(self.model, self.normalizer, kappa=float('inf'))
        if self.random_targets:
            cw_loss.forward = functools.partial(cw_loss.forward, targeted=True)
        perturbation_loss = lf.PerturbationNormLoss(lp=2)
        pert_factor = 0.0
        if self.perturbation_norm_loss is True:
            pert_factor = 0.05
        elif type(self.perturbation_norm_loss) is float:
            pert_factor = self.perturbation_norm_loss
        adv_loss = lf.RegularizedLoss({
            'cw': cw_loss,
            'pert': perturbation_loss,
        }, {
            'cw': 1.0,
            'pert': pert_factor,
        }, negate=True)

        self.pgd_attack = aa.PGD(self.model, self.normalizer,
                                 self.threat_model(), adv_loss)

        attack_params = {
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': self.lr},
            'signed': False,
            'verbose': False,
            'num_iterations': 0 if self.randomize else PGD_ITERS,
            'random_init': self.randomize,
        }
        attack_params.update(self.attack_kwargs)

        self.attack = advtrain.AdversarialAttackParameters(
            self.pgd_attack,
            1.0,
            attack_specific_params={'attack_kwargs': attack_params},
        )
        self.attack.set_gpu(False)

    def forward(self, inputs, labels):
        if self.attack is None:
            self._setup_attack()
        assert self.attack is not None

        if self.random_targets:
            return utilities.run_attack_with_random_targets(
                lambda inputs, labels: self.attack.attack(inputs, labels)[0],
                self.model,
                inputs,
                labels,
                num_classes=self.num_classes,
            )
        else:
            return self.attack.attack(inputs, labels)[0]


class UARModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        inverse_transform = InverseImagenetTransform(x.size()[-1])
        return self.model(inverse_transform(x) / 255)


class UARAttack(nn.Module):
    """
    One of the attacks from the paper "Testing Robustness Against Unforeseen
    Adversaries".
    """

    def __init__(self, model, dataset_name, attack_name, bound,
                 num_iterations=PGD_ITERS, step=None, random_targets=False,
                 randomize=False):
        super().__init__()

        assert randomize is False

        if step is None:
            step = bound / (num_iterations ** 0.5)

        self.random_targets = random_targets
        self.num_classes = DATASET_NUM_CLASSES[dataset_name]
        # if (
        #    dataset_name.startswith('mstar')
        #    or dataset_name == 'bird_or_bicycle'
        # ):
        #     dataset_name = 'imagenet'
        # elif dataset_name == 'cifar':
        #     dataset_name = 'cifar-10'

        self.model = model
        self.uar_model = UARModel(model)
        self.attack_name = attack_name
        self.bound = bound
        self.attack_fn = get_uar_attack(dataset_name, attack_name, eps=bound,
                                        n_iters=num_iterations,
                                        step_size=step, scale_each=1)
        self.attack = None

    def threat_model_contains(self, inputs, adv_inputs):
        """
        Returns a boolean tensor which indicates if each of the given
        adversarial examples given is within this attack's threat model for
        the given natural input.
        """

        if self.attack_name == 'pgd_linf':
            dist = (inputs - adv_inputs).reshape(inputs.size()[0], -1) \
                .abs().max(1)[0] * 255
        elif self.attack_name == 'pgd_l2':
            dist = (
                (inputs - adv_inputs).reshape(inputs.size()[0], -1)
                ** 2
            ).sum(1).sqrt() * 255
        elif self.attack_name == 'fw_l1':
            dist = (
                (inputs - adv_inputs).reshape(inputs.size()[0], -1)
                .abs().sum(1)
                * 255 / functools.reduce(mul, inputs.size()[1:])
            )
        else:
            raise NotImplementedError()

        return dist <= self.bound

    def forward(self, inputs, labels):
        self.uar_model.training = self.model.training

        if self.attack is None:
            self.attack = self.attack_fn()
            self.attack.transform = LambdaLayer(lambda x: x / 255)
            self.attack.inverse_transform = LambdaLayer(lambda x: x * 255)

        if self.random_targets:
            attack = lambda inputs, targets: self.attack(
                self.uar_model,
                inputs,
                targets,
                avoid_target=False,
                scale_eps=False,
            )
            adv_examples = utilities.run_attack_with_random_targets(
                attack, self.model, inputs, labels, self.num_classes,
            )
        else:
            adv_examples = self.attack(self.uar_model, inputs, labels,
                                       scale_eps=False, avoid_target=True)

        # Some UAR attacks produce NaNs, so try to get rid of them here.
        perturbations = adv_examples - inputs
        perturbations[torch.isnan(perturbations)] = 0
        return (inputs + perturbations).detach()


class LinfAttack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'mstar': 8,
                'acd': 8,
                'opensar': 8,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='pgd_linf',
            bound=bound,
            **kwargs,
        )


class L2Attack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'mstar': 0.3,
                'acd': 0.2,
                'opensar': 0.15
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='pgd_l2',
            bound=bound,
            **kwargs,
        )


class L1Attack(UARAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'mstar': 0.08,
                'acd': 0.05,
                'opensar': 0.03,
            }[dataset_name]

        super().__init__(
            model,
            dataset_name=dataset_name,
            attack_name='fw_l1',
            bound=bound,
            **kwargs,
        )



class StAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.05, **kwargs):
        kwargs.setdefault('lr', 0.01)
        super().__init__(
            model,
            threat_model=lambda: ap.ThreatModel(ap.ParameterizedXformAdv, {
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_class': st.FullSpatial,
                'use_stadv': True,
            }),
            perturbation_norm_loss=0.0025 / bound,
            **kwargs,
        )



class AutoAttack(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()

        kwargs.setdefault('verbose', False)
        # self.dataset_name=dataset_name#新加
        self.model = model
        self.kwargs = kwargs
        self.attack = None

    def forward(self, inputs, labels):
        # Necessary to initialize attack here because for parallelization
        # across multiple GPUs.
        if self.attack is None:
            try:
                # import torchattacks
                # import autoattack
                # import sys
                # sys.path.append('.')
                from autoattack import AutoAttack
                # from autoattack import AutoAttack
            except ImportError:
                raise RuntimeError(
                    'Error: unable to import autoattack. Please install the '
                    'package by running '
                    '"pip install git+git://github.com/fra31/auto-attack#egg=autoattack".'
                )
            # self.num_classes = DATASET_NUM_CLASSES[self.dataset_name]
            self.attack = AutoAttack(self.model, device=inputs.device, **self.kwargs)


        return self.attack.run_standard_evaluation(inputs, labels)


class AutoLinfAttack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'mstar': 8/255,
                'acd': 8/255,
                'opensar': 8/255,
            }[dataset_name]

        super().__init__(
            model,
            norm='Linf',
            eps=bound,
            **kwargs,
        )


class AutoL2Attack(AutoAttack):
    def __init__(self, model, dataset_name, bound=None, **kwargs):
        if bound is None:
            bound = {
                'mstar': 0.05,
                'acd': 0.05,
                'opensar': 0.03,
            }[dataset_name]

        super().__init__(
            model,
            norm='L2',
            eps=bound,
            **kwargs,
        )



def get_attack(atk,model,num_class):
    attack = None
    '''
    attacks = ["fgsm","pgd","cw","deepfool","sqa","op","vmi","hsj"]
    L0范数
    表示非0元素的个数,对抗样本中表示扰动的非0元素的个数

    L2范数
    表示各元素的平方和再开方,对抗样本中表示扰动的各元素的平方和再开平方根；针对图像数据,L2范数越小表示对抗样本人眼越难识别

    L∞范数
    表示各元素的绝对值的最大值,对抗样本中表示扰动的各元素的最大值

    '''
    if atk == 'JSMA':
        attack =  torchattacks.JSMA(model,gamma=0.01,num_class=num_class) #L0
    elif atk == 'CW':
        attack = torchattacks.CW(model,steps=10)
        # attack = torchattacks.DeepFool(model,steps=20,overshoot=0.08) #L2
    elif atk == 'FGSM':
        # attack = torchattacks.FGSM(model,eps=8/255) #Linf
        attack = torchattacks.FGSM(model,eps=4/255) #Linf
    elif atk == 'PGD':
        # attack = torchattacks.PGD(model,steps=10,eps=8/255) #Linf
        attack = torchattacks.PGD(model,steps=10,eps=4/255) #Linf
    elif atk == 'BIM':
        attack = torchattacks.BIM(model,eps=8/255,steps=2) #Linf
    elif atk == 'EADEN':
        attack = torchattacks.EADEN(model, lr=0.01, binary_search_steps=4, max_iterations=20) #L2
    elif atk == 'APGD':
        attack = torchattacks.APGD(model, norm='Linf', eps=4/255, steps=2) #Linf
    elif atk == 'SparseFool':
        attack = torchattacks.SparseFool(model, steps=10, lam=3, overshoot=0.01)

    elif atk == 'AutoLinfAttack':
        # attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, n_classes=num_class)#L2
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=4/255, n_classes=num_class)#L2
    elif atk == 'AutoL2Attack':
        # attack = torchattacks.AutoAttack(model, norm='L2', eps=8/255, n_classes=num_class)  # L2 0.05
        attack = torchattacks.AutoAttack(model, norm='L2', eps=4/255, n_classes=num_class)  # L2 0.05
    elif atk == 'sparseRS':
        attack = RSAttack(
        predict=lambda x: model_test(x.to(device)).cpu(),  # 包装模型确保输入输出设备一致
        norm='L0',
        n_queries=5000,
        eps=100,
        p_init=0.3,
        n_restarts=1,
        seed=0,
        verbose=True,
        targeted=False,
        loss='margin',
        device='cpu'  # 强制使用CPU
        )
    elif atk == 'OnePixel':
        attack = torchattacks.OnePixel(model,pixels=16,steps=50) #L0
    elif atk == 'Square':
        attack = torchattacks.Square(model,eps= 8/255, n_queries=500) #Linf
    elif atk == 'SquareL2':
        attack = torchattacks.Square(model,norm="L2",eps=2, n_queries=500) #L2
    elif atk == 'NoAttack':
        attack = NoAttack()

    else:
        raise AssertionError("attack error")
    return attack
