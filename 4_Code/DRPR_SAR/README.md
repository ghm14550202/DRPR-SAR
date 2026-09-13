# DRPR-SAR

PyTorch code and experiment artifacts for class-disentangled representation
learning and adversarially robust synthetic-aperture-radar (SAR) image
classification.

The repository is organized around a two-stage pipeline:

1. Learn a reconstruction model that separates class-relevant content from a
   residual component.
2. Train and evaluate classifiers with adversarial examples, reconstruction
   supervision, residual supervision, and knowledge distillation.

The codebase contains VQ-VAE model implementations, CNN backbones, attack utilities, pretrained classifiers, stage checkpoints, and representative evaluation logs. 

## Release contents

This checkout contains:

- SAR VQ-VAE implementations;
- ResNet, VGG, DenseNet, MobileNet, and related classifier backbones;
- local AutoAttack and `perceptual_advex` utilities;
- classifier checkpoints for MSTAR and FUSAR;
- stage-1 and stage-2 checkpoints under `results/`;
- evaluation logs and reconstruction/residual visualizations under `results_logs/`.


## Repository layout

```text
.
├── advex/                        Attack and perceptual-distance utilities
├── autoattack/                   Local AutoAttack implementation
├── classifiers_resnet/           Clean classifier checkpoints
│   ├── fusar/
│   └── mstar/
├── data/                         Dataset mount point;
├── networks/                     VAE, VQ-VAE, and classifier definitions
├── pretrained/                   Optional external pretrained weights
├── results/                      Stage checkpoints
│   ├── stage1/
│   └── stage2/
├── results_logs/                 Evaluation logs and visualizations
├── tools/                        Training and evaluation entry points
└── utils/                        Data loading, normalization, augmentation
```

## Installation

Use Python 3.8 or newer and install a matching PyTorch/torchvision pair for
your CUDA or CPU environment first. Then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

W&B is configured for offline logging by the supplied experiment scripts.
To disable logging in a surrounding environment, set:

```bash
export WANDB_MODE=disabled
```

Run commands from the repository root so that the relative `data/`, `results/`
and module paths resolve correctly.

## Dataset preparation

The loaders use `torchvision.datasets.ImageFolder`. Place each dataset under
`data/` with matching class-directory names in the training and test splits:

```text
data/
├── mstar/
│   ├── train/<class_name>/*
│   └── test/<class_name>/*
├── FUSAR/
│   ├── train/<class_name>/*
│   └── test/<class_name>/*
└── SOC_40classes/
    ├── train/<class_name>/*
    └── test/<class_name>/*
```

`ImageFolder` assigns labels in sorted class-name order, so class names and
their ordering must be consistent across splits.

## Training

### Stage 1: class-disentangled reconstruction

The main legacy stage-1 script is `tools/disentangle_SAR.py`. 
The objective combines reconstruction loss, classification loss, and a VAE
or VQ commitment term. Checkpoints are written periodically as `model_epoch*.pth`.

### Stage 1 inspection

`tools/disentangle_test.py` evaluates a stage-1 model and can save original,
reconstructed, and residual images:


### Stage 2: adversarial evaluation and defense

`tools/adv_test_SAR.py` evaluates a VAE-plus-classifier model against clean
inputs and several adversarial attacks. The command-line attack names include:

```text
NoAttack  FGSM  PGD  CW  BIM  APGD
AutoLinfAttack  AutoL2Attack  OnePixel
Square  SquareL2  SparseFool
```

Attack defaults are defined in
`perceptual_advex/attacks.py`. They should be reported together with the
results because different attacks use different norms, budgets, and numbers
of iterations.

## Checkpoints

The supplied weights preserve the filenames used by the experiments:

| Path | Description |
| --- | --- |

| Clean ResNet-50 classifier |
| `classifiers_resnet/<dataset>/*_resnet50_best.pth` |
| `classifiers_resnet/<dataset>/*_vgg_best.pth` | 
| `classifiers_resnet/<dataset>/*_densenet121_best.pth` | 

| Stage-1 checkpoint |
| `results/stage1/<dataset>/model_epoch.pth` |

| Stage-2 checkpoint |
| `results/stage2/<dataset>/model_r_epoch.pth` | 
| `results/stage2/<dataset>/robust_model_g_epoch.pth` | 
| `results/stage2/<dataset>/robust_vae_epoch.pth` | 

## Representative results

The following values are recorded in `results_logs/MSTAR_test.log` and
`results_logs/FUSAR_test.log`. They are provided as reference outputs from
the included experiments. The logs also contain per-class support, precision, class accuracy, prediction
bias, and attack progress information.
