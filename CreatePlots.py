# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:44:48 2019

@author: Thomas
"""
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# Functions to create plots
# =============================================================================
def create_plts(out_params, cross_validation, benchmark, split, root, learning_rate, num_epochs):
    split = str(split)
    if cross_validation:
        calcSTDandMEANplot(out_params, 0,1, 'loss', root, learning_rate, num_epochs)
        calcSTDandMEANplot(out_params, 3,4, 'accuracy', root, learning_rate, num_epochs)
        calcSTDandMEANplot(out_params, 5,6, 'MCC', root, learning_rate, num_epochs)
    elif benchmark:
        c = ['Others(non-Sp)', 'S', 'T', 'L'] #['I','M','O', 'S', 'T', 'L']
        plot_confusion_matrix (out_params, c, root, learning_rate, num_epochs, split, title = 'Confusion matrix of benchmark split '+split+', without normalization')
        plot_confusion_matrix (out_params, c, root, learning_rate, num_epochs, split, normalize=True, title = 'Confusion matrix of benchmark split '+split+', with normalization')
    else:        
        #------------------------------Loss------------------------------
        loss_val, loss_train, epochs, acc_val, acc_train, mcc_val, mcc_train = (np.array([x[0] for x in out_params]),
        np.array([x[1] for x in out_params]), np.array([x[2] for x in out_params]), np.array([x[3] for x in out_params]),
        np.array([x[4] for x in out_params]), np.array([x[5] for x in out_params]), np.array([x[6] for x in out_params]))
        x = np.linspace(epochs.min(),epochs.max(),500)
        loss_val, loss_train, acc_val, acc_train, mcc_val, mcc_train = (make_interp_spline(epochs, loss_val, k=3), make_interp_spline(epochs, loss_train, k=3),
        make_interp_spline(epochs, acc_val, k=3), make_interp_spline(epochs, acc_train, k=3),
        make_interp_spline(epochs, mcc_val, k=3), make_interp_spline(epochs, mcc_train, k=3))
        plt.plot(x, loss_val(x),  label='Loss of the validation data')
        plt.plot(x, loss_train(x),  label='Loss of the train data')
        plt.legend()
        plt.title('Train vs validation loss')
        plt.xlabel('Number of epochs')
        plt.ylabel('Model loss') 
        plt.savefig(root + 'Pictures\\loss_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------Accuracy------------------------------
        plt.plot(x, acc_val(x),  label='Accuracy on the validation split ' + split)
        plt.plot(x, acc_train(x),  label='Accuracy on the train data')
        plt.legend()
        plt.title('Train vs validation  accuracy' + split)
        plt.xlabel('Number of epochs')
        plt.ylabel('Model accuracy in %') 
        plt.savefig(root + 'Pictures\\acc_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------MCC------------------------------
        plt.plot(x, mcc_val(x),  label='MCC of the validation split ' + split)
        plt.plot(x, mcc_train(x),  label='MCC of the train data')
        plt.legend()
        plt.title('Train vs validation MCC')
        plt.xlabel('Number of epochs')
        plt.ylabel('Model MCC')
        plt.savefig(root + 'Pictures\\mcc_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------Confusion matrix------------------------------
        c = ['Others(non-Sp)', 'S', 'T', 'L'] #['I','M','O', 'S', 'T', 'L']
        last_entry = out_params[len(out_params)-1]
        cm_valid, cm_train = last_entry[len(last_entry)-1] , last_entry[len(last_entry)-2]
        plot_confusion_matrix (cm_train, c, root, learning_rate, num_epochs, split, title = 'Confusion matrix trainset, without normalization')
        plot_confusion_matrix (cm_valid, c, root, learning_rate, num_epochs, split, title = 'Confusion matrix of validationset split '+split+', without normalization')
        plot_confusion_matrix (cm_train, c, root, learning_rate, num_epochs, split, normalize=True, title = 'Confusion matrix trainset, with normalization')
        plot_confusion_matrix (cm_valid, c, root, learning_rate, num_epochs, split, normalize=True, title = 'Confusion matrix  split '+split+', with normalization')
    
        
def calcSTDandMEANplot(out_params, x, y, param, root, learning_rate, num_epochs):
    mean_valid = []
    std_valid = []
    mean_train = []
    std_train = []
    epochs = []
    for idx in range(len(out_params[0])):
        std_valid.append(np.std([split[idx][x] for split in out_params]))
        mean_valid.append(np.mean([split[idx][x] for split in out_params]))
        std_train.append(np.std([split[idx][y] for split in out_params]))
        mean_train.append(np.mean([split[idx][y] for split in out_params]))
        epochs.append(out_params[0][idx][2])
    mean_valid, std_valid, mean_train, std_train, epochs = (np.array(mean_valid),
    np.array(std_valid), np.array(mean_train), np.array(std_train), np.array(epochs))  
    smooth = np.linspace(epochs.min(),epochs.max(),500)
    funcx = (make_interp_spline(epochs, mean_valid, k=3))
    funcy = (make_interp_spline(epochs, mean_train, k=3))
    funcstdx = (make_interp_spline(epochs, std_valid, k=3))
    funcstdy = (make_interp_spline(epochs, std_train, k=3))
    plt.figure()
    plt.plot(smooth, funcx(smooth),  label='Standardized '+ param +' of the validation data')
    plt.fill_between(smooth, funcx(smooth)-funcstdx(smooth), funcx(smooth)+funcstdx(smooth), alpha=0.5)
    plt.plot(smooth, funcy(smooth),  label='Standardized '+ param +' of the trainings data')
    plt.fill_between(smooth, funcy(smooth)-funcstdy(smooth), funcy(smooth)+funcstdy(smooth), alpha=0.5)
    plt.legend()
    plt.title('Standardized '+ param)
    plt.xlabel('Number of epochs')
    plt.ylabel('Model '+ param) 
    plt.savefig(root + 'Pictures\\Standardized_'+ param +'_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_plot.png')
    plt.close()
    
def plot_confusion_matrix (cm, classes, root, learning_rate, num_epochs, split, normalize=False, title=None, 
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    plt.savefig(root + 'Pictures\\' + title + '_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
    plt.close()
    return ax

