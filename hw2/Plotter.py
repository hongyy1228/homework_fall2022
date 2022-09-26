import os
import tensorflow as tf
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

figsize = (5.7, 3)
export_dir = os.path.join('../image')

sns.set_theme()
sns.set_context('paper')

def get_section_results(file):
    X = []
    Y = []
    Z = []
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                Z.append(v.simple_value)
    return X, Y, Z


def read_q1_data(batch):
    full_data = pd.DataFrame()

    for folder in os.listdir('data'):
        split = folder.split('_')
        if 'CartPole-v0' in split and batch in split:
            config_list = split[split.index(batch):split.index('CartPole-v0')]
            config = '_'.join(config_list)

            logdir = os.path.join('data', folder, 'events*')
            eventfile = glob.glob(logdir)[0]

            X, Y, Z = get_section_results(eventfile)
            data = pd.DataFrame({'Iteration': range(len(X)),
                                 'Config': np.repeat(config, len(X)),
                                 'Train_EnvstepsSoFar': X,
                                 'Eval_AverageReturn': Y,
                                 'Eval_StdReturn': Z})
            data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()
            data['Eval_StdReturn'] = data['Eval_StdReturn'].ewm(alpha=0.6).mean()
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

    return full_data

data_lb = read_q1_data('lb')
data_sb = read_q1_data('sb')

