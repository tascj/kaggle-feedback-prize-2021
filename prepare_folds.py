import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

df = pd.read_csv('../data/train.csv')
sample_ids = np.array(df['id'].unique())

df['fold'] = -1
kf = KFold(n_splits=5, random_state=0, shuffle=True)
splits = kf.split(sample_ids)
for fold, (train_inds, val_inds) in enumerate(splits):
    val_ids = sample_ids[val_inds]
    df.loc[df['id'].isin(val_ids), 'fold'] = fold

print(df['fold'].value_counts())
df.to_csv('../data/dtrainval.csv', index=False)
