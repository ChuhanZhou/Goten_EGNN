# Goten EGNN

This repository contains the implementation of **GotenNet** and the proposed modified architecture for molecular property prediction on the QM9 dataset.

## Setup

### Install environment

#### Conda

```bash
conda env create -f environment.yml
conda activate env_egnn
```

#### Pip

Recommended Python version: **3.12**

```bash
pip install -r requirements.txt
```

---

## Training

### Train the reproduced GotenNet

```bash
python3 train.py \
    --ver qm9 \
    --seed $SEED \
    --label $LABEL \
    --title $TITLE
```

Example:

```bash
python3 train.py \
    --ver qm9 \
    --seed 42 \
    --label alpha \
    --title gotennet_baseline
```

### Train the modified GotenNet

```bash
python3 train.py \
    --ver qm9_my \
    --seed $SEED \
    --label $LABEL \
    --title $TITLE \
    --my_net True
```

Example:

```bash
python3 train.py \
    --ver qm9_my \
    --seed 42 \
    --label alpha \
    --title gotennet_modified \
    --my_net True
```

---

## Evaluation

To evaluate a trained model, specify the corresponding experiment title:

```bash
python3 test.py --title $TITLE
```

Example:

```bash
python3 test.py --title gotennet_modified
```

---

## Command-line Arguments

| Argument | Description                                                                                              |
|----------|----------------------------------------------------------------------------------------------------------|
| `--ver` | Dataset/configuration version (`qm9`,`qm9_s` for the reproduced model, `qm9_my` for the modified model). |
| `--seed` | Random seed.                                                                                             |
| `--label` | Target property to predict (e.g., `alpha`, `cv`, `homo`, etc.).                                          |
| `--title` | Experiment name used for saving checkpoints and logs.                                                    |
| `--my_net` | Enable the modified GotenNet architecture (`True` for the modified model).                               |

---

## Output

All checkpoints, logs, and prediction results are saved to the directories specified in [`configs/config.py`](configs/config.py).

Please modify the paths in this file if you would like to change the output locations.