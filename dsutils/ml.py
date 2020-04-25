from sklearn.metrics import log_loss, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold

import numpy as np
import matplotlib.pyplot as plt

kf = KFold(n_splits=10, shuffle=True, random_state=101)

def train_model(model, data, feats, target, **kwargs):
    model_performance = {
        'log loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1 score': []
    }

    for train_indices, test_indices in kf.split(data):
        X_train = data[feats].iloc[train_indices]
        y_train = data[target].iloc[train_indices]

        X_test = data[feats].iloc[test_indices]
        y_test = data[target].iloc[test_indices]

        model.fit(X_train, y_train)
        y_pred_ = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        model_performance['log loss'].append(log_loss(y_test, y_pred_))
        model_performance['accuracy'].append(accuracy_score(y_test, y_pred))
        model_performance['precision'].append(precision_score(y_test, y_pred))
        model_performance['recall'].append(recall_score(y_test, y_pred))
        model_performance['f1 score'].append(f1_score(y_test, y_pred))

    
    fig = plt.figure(figsize=(20, 6))

    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1)

    ax1.plot(model_performance['log loss'], label='log loss per iteration')
    ax1.plot(np.ones(10)*np.mean(model_performance['log loss']), '--', label='mean log loss')
    
    ax1.plot(model_performance['accuracy'], label='accuracy per iteration')
    ax1.plot(np.ones(10)*np.mean(model_performance['accuracy']), '--', label='mean accuracy')
    
    if 'plot_precision' in kwargs.keys() and kwargs['plot_precision'] == True:
        ax1.plot(model_performance['precision'], label='precision per iteration')
        ax1.plot(np.ones(10)*np.mean(model_performance['precision']), '--', label='mean precision')    
    
    if 'plot_recall' in kwargs.keys() and kwargs['plot_recall'] == True:
        ax1.plot(model_performance['recall'], label='recall per iteration')
        ax1.plot(np.ones(10)*np.mean(model_performance['recall']), '--', label='mean recall')    

    if 'plot_f1' in kwargs.keys() and kwargs['plot_f1'] == True:
        ax1.plot(model_performance['f1 score'], label='f1 score per iteration')
        ax1.plot(np.ones(10)*np.mean(model_performance['f1 score']), '--', label='mean f1 score')    
    
    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('fold')
    ax1.set_ylabel('value')
    ax1.set_title('Model Performance')

    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)

    ax2.bar(x=feats+['intercept'], height=np.append(model.coef_[0], model.intercept_[0]))
    ax2.grid()
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
    ax2.set_title('Model Coefficients')
    
    return model_performance, model