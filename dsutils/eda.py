import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def feature_distribution(data, feature, target):
    """
    Function that computes a overall and target class wise univariate summary of a feature 
    :param data: dataframe
    :param feature: name of the feature attribute
    :param target: name of the target attribute
    returns: A dataframe containing the computed information
    """
    description = {
        'min' : [],
        '1st' : [],
        '25th': [],
        '50th': [],
        '75th': [],
        '99th': [],
        'max' : [],
        'mean': [],
        'std' : []
    }
    if target is not None:
        for i in sorted(data[target].unique()):
            description['min'].append(round(data[data[target] == i][feature].min(), 2))
            description['1st'].append(round(data[data[target] == i][feature].quantile(0.01), 2))
            description['25th'].append(round(data[data[target] == i][feature].quantile(0.25), 2))
            description['50th'].append(round(data[data[target] == i][feature].quantile(0.50), 2))
            description['75th'].append(round(data[data[target] == i][feature].quantile(0.75), 2))
            description['99th'].append(round(data[data[target] == i][feature].quantile(0.99), 2))
            description['max'].append(round(data[data[target] == i][feature].max(), 2))
            description['mean'].append(round(data[data[target] == i][feature].mean(), 2))
            description['std'].append(round(data[data[target] == i][feature].std(), 2))
    description['min'].append(round(data[feature].min(), 2))
    description['1st'].append(round(data[feature].quantile(0.01), 2))
    description['25th'].append(round(data[feature].quantile(0.25), 2))
    description['50th'].append(round(data[feature].quantile(0.50), 2))
    description['75th'].append(round(data[feature].quantile(0.75), 2))
    description['99th'].append(round(data[feature].quantile(0.99), 2))
    description['max'].append(round(data[feature].max()))
    description['mean'].append(round(data[feature].mean(), 2))
    description['std'].append(round(data[feature].std(), 2))        
    description = pd.DataFrame(description)
    if target is not None:
        description.index = list(sorted(data[target].unique()))+ ['overall']
    else:
        description.index = ['overall']
    return description

def plot_feature_distribution(data, feature, target, labels=None, lthreshold=None, uthreshold=None):
    """
    Function that plots a overall and target class wise univariate summary of a feature 
    :param data: dataframe
    :param feature: name of the feature attribute
    :param target: name of the target attribute
    :param labels: optional labels for each target class values
    :param lthreshold: optional lower threshold (inclusive) for class wise plot
    :param uthreshold: optional upper threshold (inclusive) for class wise plot
    returns: figure and axes object of the plot
    """
    fig, ax = plt.subplots(figsize=(20,12), ncols=1, nrows=2)

    ax[0].set_title(f'Distribution of "{feature}"')
    sns.distplot(data[feature], ax=ax[0])
    ax[0].grid()

    if target is not None:
        lthreshold = data[feature] >= lthreshold if lthreshold is not None else True
        uthreshold = data[feature] <= uthreshold if uthreshold is not None else True
        labels = sorted(data[target].unique()) if labels is None else labels

        ax[1].set_title(f'Classwise Distribution of "{feature}"')

        for i in range(len(labels)):
            sns.distplot(data[(data[target] == i) & (lthreshold) & (uthreshold)][feature], ax=ax[1], label=labels[i])
        ax[1].grid()
        ax[1].legend()    
    return fig, ax