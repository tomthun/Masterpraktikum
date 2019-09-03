# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:36:46 2019

@author: Thomas
"""
import os
os.chdir('C:\\Users\\Thomas\\Documents\\Uni_masters\\Masterpraktikum')
from CustomDataset import CustomDataset
from CNN import SimpleCNN
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import time
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from CreatePlots import create_plts
import torch as nn

timer = time.time()
splits = 5 
num_classes = 4
#--------------- parameterize hyperparameters ---------------
nn.manual_seed(10)
root = 'C:\\Users\\Thomas\\Documents\\Uni_masters\\MasterPrak_Data\\'
params = {'batch_size': 200,
          'shuffle': True,
          'num_workers': 0}
num_epochs = 501
learning_rate = 5e-4
weights = [0.05, 0.95, 0.99, 0.98]
dev = torch.device('cuda')
#dev = torch.device('cpu')
class_weights = torch.FloatTensor(weights).to(dev)
printafterepoch = 50
#--------------- Cross Validation ---------------
cross_validation = False
#--------------- Parameterize grid search here ---------------
gridsearch = False
all_num_epochs = (50,100,150,200)
all_learning_rate = (1e-2, 1e-3, 1e-4, 5e-4 ,1e-5)
#--------------- Benchmark ---------------
benchmark = True
normal_run = False
#---Selected split to benchmark/validate upon (0-4, !must not be the same!)---
selected_split = 0
benchmark_split = 4
# =============================================================================
# Main functions
# =============================================================================
def cross_validate():
    # do crossvalidation
    params_list = []
    print('Starting cross-validation...')
    for split in range(splits):
        if split == benchmark_split:
            print('Skipping benchmark split...')
            continue
        out = main(split, False)
        params_list.append(out)
    create_plts(params_list, cross_validation, False, split, root, learning_rate, num_epochs)  
    return params_list

def main(split,benchmark):
    # create data folders if non-existent
    if not os.path.isdir(root + 'Pictures'):
        os.mkdir(root + 'Pictures')
    elif not os.path.isdir(root + 'pickled_files'):
        os.mkdir(root + 'pickled_files')
    if benchmark:
        print('Benchmarkset is:', benchmark_split)
        try:
            model = torch.load(root + 'model.pickle')
            print('Your existing model will be benchmarked')
            bench_data, bench_labels, bench_orga = de_serializeBenchmark(benchmark_split)
            bench_dataset = CustomDataset(bench_data, bench_labels, bench_orga)
            bench_loader = DataLoader(bench_dataset, **params)
            result, true_mcc, loss_ave, cm , mcc_orga, cm_orga, label_predicted_batch = validate(bench_loader, model, dev)
            print('Confusion matrix is:\n', cm)
            createcompfile(root,label_predicted_batch)
            create_plts(cm, cross_validation, benchmark, benchmark_split, root, learning_rate, num_epochs, mcc_orga = mcc_orga, cm_orga = cm_orga)
        except:
            print('No model found for benchmarking! Start a new run with benchmark = False!')
    else:
        print('Validationset is:', split, 'Benchmarkset is:', benchmark_split)
        # train on the GPU or on the CPU, if a GPU is not available
        train_data, train_labels, validation_data, validation_labels, info, train_orga, validation_orga = de_serializeInput(split)
        train_dataset = CustomDataset(train_data, train_labels, train_orga)
        validation_dataset = CustomDataset(validation_data, validation_labels, validation_orga)
        train_loader = DataLoader(train_dataset, **params)
        validation_loader = DataLoader(validation_dataset, **params)
        model = SimpleCNN(num_classes)
        model = model.to(dev)
        model, out_params, label_predicted_batch = train(model, train_loader, validation_loader, 
                                  num_epochs, learning_rate, dev)
        if normal_run and not cross_validation and not gridsearch:
            create_plts(out_params, cross_validation, benchmark ,split, root, learning_rate, num_epochs)
        torch.save(model, root + 'model.pickle')
        return out_params
# =============================================================================
# Load / preprocess data
# =============================================================================
def de_serializeInput(validation_split):    
    all_features, info = loaddata('signalP4.npz', 'train_set.fasta')
    train_data,train_labels,train_orga = [],[],[]
    try:       
        print('Loading pickled files...')
        for split in range(splits):
            split_data = pickle.load(open(root+"pickled_files\\split_"+str(split)+"_data.pickle", "rb"))
            split_labels = pickle.load(open(root+"pickled_files\\split_"+str(split)+"_labels.pickle", "rb"))
            split_orga  = pickle.load(open(root+"pickled_files\\split_"+str(split)+"_orga.pickle", "rb"))
            if split == benchmark_split:
                continue
            elif split == validation_split:
                validation_data,validation_labels, validation_orga = split_data, split_labels, split_orga
            else:
                train_data.extend(split_data)
                train_labels.extend(split_labels)
                train_orga.extend(split_orga)
        print('Done!')
    except (OSError, IOError):     
        print('Pickled files not found!\nCreating new train/validation dataset...')
        for split in range(splits):           
            split_keys = selectTestTrainSplit(info,split)
            split_data, split_labels, split_orga = createDataVectors(info,all_features,split_keys)
            pickle.dump(split_data, open( root+"pickled_files\\split_"+str(split)+"_data.pickle", "wb" ))
            pickle.dump(split_labels, open( root+"pickled_files\\split_"+str(split)+"_labels.pickle", "wb" ))
            pickle.dump(split_orga, open( root+"pickled_files\\split_"+str(split)+"_orga.pickle", "wb" ))
            if split == benchmark_split:
                continue
            elif split == validation_split:
                validation_data,validation_labels, validation_orga = split_data, split_labels, split_orga
            else:
                train_data.extend(split_data)
                train_labels.extend(split_labels)
                train_orga.extend(split_orga)
        print('Saved and Done!')
    return train_data,train_labels,validation_data,validation_labels,info, train_orga, validation_orga

def  de_serializeBenchmark(bench_split):
    all_features, info = loaddata('signalP4.npz', 'benchmark_set.fasta')
    try:       
        print('Loading pickled benchmark files...')        
        bench_data = pickle.load(open(root+"pickled_files\\bench_"+str(bench_split)+"_data.pickle", "rb"))
        bench_labels  = pickle.load(open(root+"pickled_files\\bench_"+str(bench_split)+"_labels.pickle", "rb"))
        bench_orga  = pickle.load(open(root+"pickled_files\\bench_"+str(bench_split)+"_orga.pickle", "rb"))
        print('Done!')
    except (OSError, IOError):     
        print('Pickled files not found!\nCreating new benchmark dataset...')
        for split in range(splits):           
            split_keys = selectTestTrainSplit(info,split)
            split_data, split_labels, split_orga = createDataVectors(info,all_features,split_keys)
            pickle.dump(split_data, open( root+"pickled_files\\bench_"+str(split)+"_data.pickle", "wb" ))
            pickle.dump(split_labels, open( root+"pickled_files\\bench_"+str(split)+"_labels.pickle", "wb" ))
            pickle.dump(split_orga, open( root+"pickled_files\\bench_"+str(split)+"_orga.pickle", "wb" ))
            if split == bench_split:
                bench_data, bench_labels, bench_orga = split_data, split_labels, split_orga
        print('Saved and Done!')
    return bench_data, bench_labels, bench_orga

def loaddata (data_name , training_name):
    train_data = open(root+training_name, 'r') 
    train_data = train_data.read().split('\n')
    tmp = np.load(root+data_name)
    info = {}
    header = train_data[0].split('|')[0].replace('>','')
    org = train_data[0].split('|')[1]
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
            info[header] = [signalp, partition,seq,sig,sigbin,lenprot,org]
        else:
            # padding of proteins < 70 aminoacids
            lenprot = len(seq)
            [sigbin.append(-100) for x in range (70 - lenprot)]
            info[header] = [signalp, partition,seq,sig,sigbin,lenprot,org]
        seq = train_data[count+1]
        sig = train_data[count+2]
#         sigbin = list(map(int,sig.replace('I','0').replace('M','1').replace('O','2')
#             .replace('S','3').replace('T','4').replace('L','5')))
        sigbin = list(map(int,sig.replace('I','0').replace('M','0').replace('O','0')
             .replace('S','1').replace('T','2').replace('L','3')))
        header = train_data[count].split('|')[0].replace('>','')    
        org = train_data[count].split('|')[1]
        signalp = train_data[count].split('|')[2]
        partition = train_data[count].split('|')[3]
        count += 3
    # remove invalid Proteinidentifiers (which changed over time)
    for e in (set(list(info.keys()))-set(tmp.files)):
        info.pop(e)     
    return tmp, info

def createDataVectors(info, all_features, keys):
    data = []
    label = []
    orga = []
    for key in keys:
        lenprot = info[key][5]
        label.append(info[key][4])
        orga.append(info[key][6])
        if lenprot < 70:
            feat = all_features[key][:lenprot]
            result = np.zeros([70,1024])
            result[:feat.shape[0], :feat.shape[1]] = feat
            data.append(result)
        else:
            data.append(all_features[key][:70])
    return data, label, orga

def selectTestTrainSplit(train_data,x):
    split = [key  for (key, value) in train_data.items() if value[1] == str(x)]
    return split

def createcompfile(root, label_predicted_batch):
    f = open(root+"comparison.txt","w+")
    csdiff, gaps = cleavagediff(label_predicted_batch) 
    f.write("Mean residue cleavage residue deviation of predicted to true label: " + str(csdiff) +
            "\nGaps have been found at protein predictions: "+ str(gaps) + "\n")
    for i in range(len(label_predicted_batch[0])):
        f.write("Protein "+ str(i)+ "\n")
        f.write("True labels: " + str(label_predicted_batch[0][i].tolist()))
        f.write("\nPred labels: " + str(label_predicted_batch[1][i].astype(int).tolist()) + "\n")
    f.close()

def cleavagediff(label_predicted_batch):
    csdiff = 0
    gaps = []
    gapstr = []
    for i in range (len(label_predicted_batch[0])):
        truth, prediction = label_predicted_batch[0][i], label_predicted_batch[1][i]
        if (truth != prediction).any():
            csdiff += abs(truth[truth != 0].size - prediction[prediction != 0].size)
        gap = containsgap(prediction)
        gapsi = containsgap(truth)
        if gap:
            gaps.append(i)
        if gapsi:
            gapstr.append(i)
    csdiff = csdiff/len(label_predicted_batch[0]) 
    if len(gaps) > 0: 
        print('The prediction contains gaps at : ' + str(gaps))      
    if len(gapstr) > 0: 
        print('The true labels contain gaps at : ' + str(gapstr))
    else: print ('True labels have no gaps')
    return csdiff, gaps

def containsgap(prediction):
    gap = False
    x = (prediction == 0)
    x = np.where(x[:-1] != x[1:])[0].size
    if x > 1:
        gap = True
    return gap

# =============================================================================
# Functions for training/validation
# =============================================================================
def calcClassImbalance(info):
#    calculate class imbalance of the dataset NOT USED AT THE MOMENT
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

def orgaBatch (labels, predicted, orga, predicted_batch, labels_batch, label_predicted_batch):
    #to do: apply in validate
    predicted, labels = predicted.to('cpu').numpy(), labels.to('cpu').numpy()
    for x in range(len(labels)):
        if orga[x] == 'ARCHAEA':
            predicted_batch[0].extend(predicted[x])
            labels_batch[0].extend(labels[x])
        elif orga[x] == 'EUKARYA':            
            predicted_batch[1].extend(predicted[x])
            labels_batch[1].extend(labels[x])
        elif orga[x] == 'NEGATIVE':
            predicted_batch[2].extend(predicted[x])
            labels_batch[2].extend(labels[x])
        elif orga[x] == 'POSITIVE':
            predicted_batch[3].extend(predicted[x])
            labels_batch[3].extend(labels[x])
        label_predicted_batch[0].append(labels[x])
        label_predicted_batch[1].append(predicted[x])
    return predicted_batch, labels_batch, label_predicted_batch

def calcMCCbatch (labels_batch, predicted_batch):
#   calculate MCC over given batches of an epoch in training/validation 
#   [0]:Archea, [1]:Eukaryot, [2]:Gram negative, [3]:Gram positive  
    x = sum(predicted_batch, [])
    y = sum(labels_batch,[])
    mcc = metrics.matthews_corrcoef(x, y)
    cm = confusion_matrix(x, y,  [0, 1, 2, 3]) #[0, 1, 2, 3, 4, 5])    
    return mcc,cm

def calcMCCorga(labels_batch, predicted_batch):
#   calculate MCC over given batches of an epoch in training/validation 
#   [0]:Archea, [1]:Eukaryot, [2]:Gram negative, [3]:Gram positive 
    mcc_list, cm_list = [],[]
    for x in range(len(labels_batch)):   
        mcc = metrics.matthews_corrcoef(predicted_batch[x], labels_batch[x])
        cm = confusion_matrix(predicted_batch[x], labels_batch[x],  [0, 1, 2, 3]) #[0, 1, 2, 3, 4, 5])  
        mcc_list.append(mcc)
        cm_list.append(cm)
    return mcc_list, cm_list

def train(model, train_loader, validation_loader, num_epochs, learning_rate, dev):
    print('Starting to learn...')
    total_step = len(train_loader)
    predicted_batch = [[],[],[],[]] # [0]:Archea, [1]:Eukaryot, [2]:Gram negative, [3]:Gram positive
    labels_batch= [[],[],[],[]]
    label_predicted_batch = [[],[]]
    out_params = []
    criterion = torch.nn.CrossEntropyLoss(weight = class_weights, ignore_index = -100, reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        loss_train_list = []
        correct = 0
        total = 0 
        for i, (train, labels, mask, orga) in enumerate(train_loader):
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
                predicted_batch, labels_batch, label_predicted_batch = orgaBatch(labels, predicted, orga, predicted_batch, labels_batch, label_predicted_batch)
                loss_train_list.append(loss.item())  
                
        # and print the results
        if (epoch%printafterepoch) == 0:            
            mcc_train, cm_train = calcMCCbatch(labels_batch, predicted_batch)
            acc_train = (correct / total)*100
            loss_ave = sum(loss_train_list)/len(loss_train_list)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, MCC: {:.2f}'
                  .format(epoch+1, num_epochs, i + 1, total_step, loss_ave,
                          acc_train, mcc_train))
            acc_valid, mcc_valid, loss_valid, cm_valid, mcc_orga, cm_orga, label_predicted_batch_val = validate(validation_loader, model, dev, criterion = criterion)
            out_params.append((loss_valid, loss_ave, epoch, acc_valid, acc_train, mcc_valid, mcc_train , mcc_orga, cm_orga, cm_train, cm_valid))
    # check overfitting
    print('Best validation loss:', min(out_params)[0]  ,' at epoch:', min(out_params)[2])
    return model, out_params, label_predicted_batch

def validate(validation_loader, model, dev, criterion = None):
    with torch.no_grad():   
        correct = 0
        total = 0
        predicted_batch = [[],[],[],[]] # [0]:Archea, [1]:Eukaryot, [2]:Gram negative, [3]:Gram positive
        labels_batch= [[],[],[],[]]
        label_predicted_batch = [[],[]]
        loss_list = []
        for validation, labels, mask, orga in validation_loader:
            # preprocess outputs to correct format (1024*70*1)
            validation, labels, mask = validation.to(dev), labels.to(dev), mask.to(dev)
            outputs = model(validation.unsqueeze(3))
            outputs.squeeze_()
            outputs = outputs.permute(2,0,1)
            # apply conditional random field and decode via Vertibri algorithm
            loss = -model.crf(outputs, labels.permute(1,0), mask = mask.permute(1,0))
            predicted = nn.Tensor(model.crf.decode(outputs)).cuda()
            #loss = criterion(outputs.squeeze_(), labels)
            #_, predicted = torch.max(outputs.data, 1)
            # calculate quality measurements
            correct += (predicted == labels.float()).sum().item()    
            total = total + (labels.size(0) * labels.size(1))
            result = ((correct / total) * 100) 
            predicted_batch, labels_batch, label_predicted_batch = orgaBatch(labels, predicted, orga, predicted_batch, labels_batch, label_predicted_batch)
            loss_list.append(loss.item())         
        mcc, cm = calcMCCbatch(labels_batch, predicted_batch)
        mcc_orga, cm_orga = calcMCCorga(labels_batch, predicted_batch)
        loss_ave = sum(loss_list)/len(loss_list)
        print('Accuracy of the model on the validation proteins is: {:.2f}%, Loss:{:.3f} and MCC is: {:.2f}'.format(result,loss_ave,mcc))
    return result, mcc, loss_ave, cm,  mcc_orga, cm_orga, label_predicted_batch
# =============================================================================
# Execute when running script
# =============================================================================
if __name__ == "__main__":
    if selected_split == benchmark_split and benchmark:
        try: raise SystemExit
        except: print('Benchmark and validation split cannot be the same when doing a normal run with benchmarking because of continous biased evaluation.')
    if cross_validation and not normal_run:
        out = cross_validate() 
    else: print('Disable normal run to do cross validation!')
    if gridsearch :     
        cross_valid_params = []
        for y in range(len(all_learning_rate)):
            learning_rate = all_learning_rate[y]
            out = cross_validate()
            cross_valid_params.append(out)
    if normal_run:
        cross_validation, gridsearch = False, False
        out = main(selected_split, False)
    if benchmark:
        main(benchmark_split, benchmark)
    print("Runtime: ", time.time() - timer)