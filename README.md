# Feedback Prize - Evaluating Student Writing

https://www.kaggle.com/c/feedback-prize-2021

## Environment setup

Build docker image

```
bash .dev_scripts/build.sh
```

Set env variables

```
export DATA_DIR="/path/to/data"
export CODE_DIR="/path/to/this/repo"
```

Start a docker container
```
bash .dev_scripts/start.sh all
```

## Data preparation

1. Download competition data from Kaggle
2. Download LIVECell dataset from https://github.com/sartorius-research/LIVECell (we didn't use the data provided by Kaggle)
3. Unzip the files as follows

```
├── train
├── test
└── train.csv
```

Start a docker container and run the following commands

```
python prepare_folds.py
```

The results should look like the 

```
├── train
├── test
├── train.csv
└── dtrainval.csv
```

## Training

Start a docker container and run the following commands for training

```
# train
python train.py configs/deberta_large_fold0.py

# SWA
python swa.py work_dirs/deberta_large_fold0 2 6
```

## Inference

An example of inference is shared in the following Kaggle notebook

https://www.kaggle.com/code/tascj0/a-text-span-detector

Checkpoints could be found in the following Kaggle dataset

https://www.kaggle.com/datasets/tascj0/feedback-checkpoints
