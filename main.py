# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:36:46 2019

@author: Thomas
"""
from scipy.interpolate import make_interp_spline
from CustomDataset import CustomDataset
from CNN import SimpleCNN
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from ConfusionMatrix import plot_confusion_matrix
import torch as nn
timer = time.time()

#--------------- parameterize hyperparameters ---------------
nn.manual_seed(10)
root = 'C:\\Users\\Thomas\\Documents\\Uni_masters\\MasterPrak_Data\\'
params = {'batch_size': 200,
          'shuffle': True,
          'num_workers': 0}
num_epochs = 5
learning_rate = 5e-4
splits = 5 
weights = [0.05, 0.95, 0.99, 0.98]
dev = torch.device('cuda')
#dev = torch.device('cpu')
class_weights = torch.FloatTensor(weights).to(dev)
printafterepoch = 100
num_classes = 4
#--------------- Cross Validation ---------------
cross_validation = False

#--------------- Parameterize grid search here ---------------
grid_search = False
all_num_epochs = (50,100,150,200)
all_learning_rate = (1e-2, 1e-3, 1e-4, 5e-4 ,1e-5)

#--------------- Benchmark ---------------
benchmark = True
#--------------- selected split to benchmark/validate upon -------------------
selected_split = 4

def cross_validate():
     print('to do')
#    acc_list = [] 
#    epoch_list = []
#    print('Starting cross-validation...')
#    for split in range(splits):
#       for x in range(len(all_num_epochs)):
#           num_epochs = all_num_epochs[x]
#           for y in range(len(all_learning_rate)):
#               learning_rate = all_learning_rate[y]
#               main(str(split),class_weights)
#    acc, epoch, out = main(str(split), class_weights)
#    acc_list.append(acc)
#    epoch_list.append(epoch)
#    if (epoch/num_epochs) == 1:
#        print('No overfitting!')
#    else:
#        print('Overfitting after epoch:', epoch,'!')
#    print('Mean of the accuracy:', (sum(acc_list)/len(acc_list)))     
#    return acc_list, epoch_list

def main(split,benchmark):
    # create data folders if non-existent
    if not os.path.isdir(root + 'Pictures'):
        os.mkdir(root + 'Pictures')
    elif not os.path.isdir(root + 'pickled_files'):
        os.mkdir(root + 'pickled_files')
    if benchmark:
        try:
            model = torch.load(root + 'model.pickle')
            print('Your existing model will be benchmarked')
            bench_data, bench_labels = de_serializeBenchmark(split)
            bench_dataset = CustomDataset(bench_data, bench_labels)
            bench_loader = DataLoader(bench_dataset, **params)
            result, true_mcc, loss_ave, cm_valid  = validate(bench_loader, model, dev)
            print('Confusion matrix is:\n', cm_valid)
        except:
            print('No model found for benchmarking! Start a new run with benchmark = False!')
    else:
        print('Validationset is:', split)
        # train on the GPU or on the CPU, if a GPU is not available
        train_data, train_labels, validation_data, validation_labels, info = de_serializeInput(split)
        train_dataset = CustomDataset(train_data,train_labels)
        validation_dataset = CustomDataset(validation_data,validation_labels)
        train_loader = DataLoader(train_dataset, **params)
        validation_loader = DataLoader(validation_dataset, **params)
        model = SimpleCNN(num_classes)
        model = model.to(dev)
        model, out_params = train(model, train_loader, validation_loader, 
                                  num_epochs, learning_rate, dev)
        best_val_acc = min(out_params)[3]
        best_epoch = min(out_params)[2]
        create_plts(out_params, split)
        torch.save(model, root + 'model.pickle')
        return best_val_acc, best_epoch, out_params

def de_serializeInput(validation_split):    
    all_features, info = loaddata('signalP4.npz', 'train_set.fasta')
    train_data,train_labels,validation_data,validation_labels = [],[],[],[]
    try:       
        print('Loading pickled files...')
        for split in range(splits):
            split_data = pickle.load(open(root+"pickled_files\\split_"+str(split)+"_data.pickle", "rb"))
            split_labels = pickle.load(open(root+"pickled_files\\split_"+str(split)+"_labels.pickle", "rb"))
            if split == validation_split:
                validation_data,validation_labels = split_data, split_labels
            else:
                train_data.extend(split_data)
                train_labels.extend(split_labels)
        print('Done!')
    except (OSError, IOError):     
        print('Pickled files not found!\nCreating new train/validation dataset...')
        for split in range(splits):           
            split_keys = selectTestTrainSplit(info,split)
            split_data, split_labels = createDataVectors(info,all_features,split_keys)
            pickle.dump(split_data, open( root+"pickled_files\\split_"+str(split)+"_data.pickle", "wb" ))
            pickle.dump(split_labels, open( root+"pickled_files\\split_"+str(split)+"_labels.pickle", "wb" ))
            if split == validation_split:
                validation_data,validation_labels = split_data, split_labels
            else:
                train_data.extend(split_data)
                train_labels.extend(split_labels)
        print('Saved and Done!')
    return train_data,train_labels,validation_data,validation_labels,info

def  de_serializeBenchmark(bench_split):
    all_features, info = loaddata('signalP4.npz', 'benchmark_set.fasta')
    bench_data, bench_labels = [],[]
    try:       
        print('Loading pickled benchmark files...')        
        bench_data = pickle.load(open(root+"pickled_files\\bench_"+str(bench_split)+"_data.pickle", "rb"))
        bench_labels  = pickle.load(open(root+"pickled_files\\bench_"+str(bench_split)+"_labels.pickle", "rb"))
        print('Done!')
    except (OSError, IOError):     
        print('Pickled files not found!\nCreating new benchmark dataset...')
        for split in range(splits):           
            split_keys = selectTestTrainSplit(info,split)
            split_data, split_labels = createDataVectors(info,all_features,split_keys)
            pickle.dump(split_data, open( root+"pickled_files\\bench_"+str(split)+"_data.pickle", "wb" ))
            pickle.dump(split_labels, open( root+"pickled_files\\bench_"+str(split)+"_labels.pickle", "wb" ))
            if split == bench_split:
                bench_data, bench_labels = split_data, split_labels
        print('Saved and Done!')
    return bench_data, bench_labels

def loaddata (data_name , training_name):
    train_data = open(root+training_name, 'r') 
    train_data = train_data.read().split('\n')
    tmp = np.load(root+data_name)
    info = {}
    header = train_data[0].split('|')[0].replace('>','')
    signalp = train_data[0].split('|')[2]
    partition = train_data[0].split('|')[3]
    seq = train_data[1]
    sig = train_data[2]
#    sigbin = list(map(int,sig.replace('I','0').replace('M','1').replace('O','2')
#                .replace('S','3').replace('T','4').replace('L','5')))
    sigbin = list(map(int,sig.replace('I','0').replace('M','0').replace('O','0')
             .replace('S','1').replace('T','2').replace('L','3')))
    count = 3
    for x in range(int((len(train_data)-4)/3)):
        lenprot = 70
        if (len(seq) == lenprot):
            info[header] = [signalp, partition,seq,sig,sigbin,lenprot]
        else:
            #remove # to include shorter proteins
            lenprot = len(seq)
            [sigbin.append(-100) for x in range (70 - lenprot)]
            info[header] = [signalp, partition,seq,sig,sigbin,lenprot]
        seq = train_data[count+1]
        sig = train_data[count+2]
#         sigbin = list(map(int,sig.replace('I','0').replace('M','1').replace('O','2')
#             .replace('S','3').replace('T','4').replace('L','5')))
        sigbin = list(map(int,sig.replace('I','0').replace('M','0').replace('O','0')
             .replace('S','1').replace('T','2').replace('L','3')))
        header = train_data[count].split('|')[0].replace('>','')    
        signalp = train_data[count].split('|')[2]
        partition = train_data[count].split('|')[3]
        count += 3
    # remove invalid Proteinidentifiers
    for e in (set(list(info.keys()))-set(tmp.files)):
        info.pop(e)     
    return tmp, info

def createDataVectors(info, all_features, keys):
    data = []
    label = []
    for key in keys:
        lenprot = info[key][5]
        label.append(info[key][4])
        if lenprot < 70:
#           remove continue and # to include shorter proteins          
            feat = all_features[key][:lenprot]
            result = np.zeros([70,1024])
            result[:feat.shape[0], :feat.shape[1]] = feat
            data.append(result)
        else:
            data.append(all_features[key][:70])
    return data,label

def selectTestTrainSplit(train_data,x):
    split = [key  for (key, value) in train_data.items() if value[1] == str(x)]
    return split 

def calcClassImbalance(info):
    counts = [0,0,0,0,0,0]
    for x in info:
        classes = info[x][3]
        counts[0] = counts[0] + classes.count('I')
        counts[1] = counts[1] + classes.count('M')
        counts[2] = counts[2] + classes.count('O')
        counts[3] = counts[3] + classes.count('S')
        counts[4] = counts[4] + classes.count('T')
        counts[5] = counts[5] + classes.count('L')
    return counts

def calcMCCbatch (labels, predicted):
    predicted, labels = predicted.to('cpu').numpy(), labels.to('cpu').numpy()
    predicted_batch = []
    labels_batch = []
    cm = 0
    for x in range(len(labels)):
        predicted_batch.extend(predicted[x])
        labels_batch.extend(labels[x])
        cm += confusion_matrix(labels[x], predicted[x],  [0, 1, 2, 3]) #[0, 1, 2, 3, 4, 5])     
    mcc = metrics.matthews_corrcoef(predicted_batch, labels_batch)
    return mcc,cm

def train(model, train_loader, validation_loader, num_epochs, learning_rate, dev):
    print('Starting to learn...')
    total_step = len(train_loader)
    out_params = []
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights, ignore_index = -100, reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        mcc_train_sum =  []
        loss_train_list = []
        cm_train = 0
        correct = 0
        total = 0 
        for i, (train, labels, mask) in enumerate(train_loader):
            # Run the forward pass
            train, labels, mask = train.to(dev), labels.to(dev), mask.to(dev)
            outputs = model(train.unsqueeze(3))
            outputs = outputs.squeeze_()
            outputs = outputs.permute(2,0,1)  
            #loss = criterion(outputs, labels)
            loss = -model.crf(outputs, labels.permute(1,0), mask = mask.permute(1,0))
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Track the accuracy, mcc and cm                    
            if (epoch%printafterepoch) == 0: 
                total += labels.size(0)* labels.size(1)
                #_, predicted = torch.max(outputs.data, 1)                
                #predicted = predicted.squeeze_()
                predicted = nn.Tensor(model.crf.decode(outputs)).cuda()
                correct += (predicted == labels.float()).sum().item()              
                mcc_train, cm = calcMCCbatch(labels, predicted)
                cm_train += cm
                mcc_train_sum.append(mcc_train)
                loss_train_list.append(loss.item())  
                
        # and print the results
        if (epoch%printafterepoch) == 0:
            acc_train = (correct / total)*100
            loss_ave = sum(loss_train_list)/len(loss_train_list)
            mcc_ave = sum(mcc_train_sum)/len(mcc_train_sum)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, MCC: {:.2f}'
                  .format(epoch+1, num_epochs, i + 1, total_step, loss_ave,
                          acc_train, mcc_ave))
            acc_valid, mcc_val, loss_valid, cm_valid = validate(validation_loader, model, dev, criterion = criterion)
            out_params.append((loss_valid, loss_ave, epoch, acc_valid, acc_train, mcc_val, mcc_ave , cm_train, cm_valid))
    # check overfitting
    print('Best validation loss:', min(out_params)[0]  ,' at epoch:', min(out_params)[2])
    return model, out_params

def validate(validation_loader, model, dev, criterion = None):
    with torch.no_grad():
        correct = 0
        total = 0
        mcc_sum = []
        cm_valid = 0
        loss_list = []
        for validation, labels, mask in validation_loader:
            validation, labels, mask = validation.to(dev), labels.to(dev), mask.to(dev)
            outputs = model(validation.unsqueeze(3))
            outputs.squeeze_()
            outputs = outputs.permute(2,0,1)
            loss = -model.crf(outputs, labels.permute(1,0), mask = mask.permute(1,0))
            predicted = nn.Tensor(model.crf.decode(outputs)).cuda()
            #loss = criterion(outputs.squeeze_(), labels)
            #_, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.float()).sum().item()    
            total = total + (labels.size(0) * labels.size(1))
            result = ((correct / total) * 100)        
            mcc, cm = calcMCCbatch(labels, predicted)
            cm_valid += cm 
            mcc_sum.append(mcc)
            loss_list.append(loss.item())
        true_mcc = sum(mcc_sum)/len(mcc_sum)
        loss_ave = sum(loss_list)/len(loss_list)
        print('Accuracy of the model on the validation proteins is: {:.2f}%, Loss:{:.3f} and MCC is: {:.2f}'.format(result,loss_ave,true_mcc))
    return result, true_mcc, loss_ave, cm_valid

def create_plts(out_params, split):
    split = str(split)
    if cross_validation:
        print('todo')
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
        plt.title('Train vs validation loss of split ' + split)
        plt.xlabel('Number of epochs')
        plt.ylabel('Model loss') 
        plt.savefig(root + 'Pictures\\loss_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------Accuracy------------------------------
        plt.plot(x, acc_val(x),  label='Accuracy on the validation data')
        plt.plot(x, acc_train(x),  label='Accuracy on the train data')
        plt.legend()
        plt.title('Accuracy of split ' + split)
        plt.xlabel('Number of epochs')
        plt.ylabel('Model accuracy in %') 
        plt.savefig(root + 'Pictures\\acc_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------MCC------------------------------
        plt.plot(x, mcc_val(x),  label='MCC of the validation data')
        plt.plot(x, mcc_train(x),  label='MCC of the train data')
        plt.legend()
        plt.title('MCC of split ' + split)
        plt.xlabel('Number of epochs')
        plt.ylabel('Model MCC')
        plt.savefig(root + 'Pictures\\mcc_plot_lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_split_'+split+'.png')
        plt.close()
        #------------------------------Confusion matrix------------------------------
        c = ['Others(non-Sp)', 'S', 'T', 'L'] #['I','M','O', 'S', 'T', 'L']
        last_entry = out_params[len(out_params)-1]
        cm_valid, cm_train = last_entry[len(last_entry)-1] , last_entry[len(last_entry)-2]
        plot_confusion_matrix (cm_train, c, root, learning_rate, num_epochs, split, title = 'Confusion matrix trainset, without normalization')
        plot_confusion_matrix (cm_valid, c, root, learning_rate, num_epochs, split, title = 'Confusion matrix validationset, without normalization')
        plot_confusion_matrix (cm_train, c, root, learning_rate, num_epochs, split, normalize=True, title = 'Confusion matrix trainset, with normalization')
        plot_confusion_matrix (cm_valid, c, root, learning_rate, num_epochs, split, normalize=True, title = 'Confusion matrix validationset, with normalization')

if __name__ == "__main__":
    if cross_validation:
        acc_list, epoch_list = cross_validate() 
    else:
        main(selected_split, benchmark)
        
    print("Runtime: ", time.time() - timer)