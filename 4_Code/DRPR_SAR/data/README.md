# Dataset mount point

Dataset files are intentionally kept outside this source bundle. Put them
under this directory using the expected split structure:

```text
data/
├── mstar/
│   ├── train/<class_name>/*
│   └── test/<class_name>/*
├── FUSAR/
│   ├── train/<class_name>/*
│   └── test/<class_name>/*
└── ATRNet-STAR/
    ├── train/<class_name>/*
    └── test/<class_name>/*
```

The transformer loader uses `DATA_ROOT` and accepts an explicit
`--data_path`. Dataset images, split lists, and metadata are not generated or
copied by the organization step.
