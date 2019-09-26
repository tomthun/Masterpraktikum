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
from CreatePlots import create_plts
import torch as nn

timer = time.time()
splits = 5 
num_classes = 4
#--------------- parameterize hyperparameters ---------------
nn.manual_seed(10)
root = 'C:\\Users\\Thomas\\Documents\\Uni_masters\\MasterPrak_Data\\'
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 0}
num_epochs = 9
learning_rate = 1e-3
weights = [7.554062537062119e-07,
 1.2681182393446364e-05,
 6.0745960393633824e-05,
 3.161855376735068e-05] # 1/occurence(type of residue) see calcClassimbalance
dev = torch.device('cuda')
#dev = torch.device('cpu')
class_weights = torch.FloatTensor(weights).to(dev) # only for Cross entropy loss
printafterepoch = 4
#--------------- Disable/Enable the addition of a crf ---------------
no_crf = False
#--------------- Cross Validation ---------------
cross_validation = False
benchmarked_cross_validation = True
#--------------- Parameterize grid search here ---------------
gridsearch = False
all_num_epochs = [21,23,25,27]
all_learning_rate = (1e-3, 1e-4, 5e-4 ,1e-5)
#--------------- Benchmark ---------------
benchmark = False
normal_run = False
#---Selected split to benchmark/validate upon (0-4, !must not be the same!)---
selected_split = 0
benchmark_split = 1
# =============================================================================
# Main functions
# =============================================================================
def cross_validate():
    # do crossvalidation
    params_list, params_listbench = [],[]
    labels = []
    predictions = []
    print('Starting cross-validation...')
    for split in range(splits):
        if split == benchmark_split:
            print('Skipping benchmark split...')
            continue
        out = main(split, benchmark_split, False)
        outbench = main(split, benchmark_split, True)
        labels += outbench[8][0]
        predictions += outbench[8][1]
        params_list.append(out)
        params_listbench.append(outbench)
    return params_list, params_listbench, labels, predictions

def cross_benchmark():
    params_list = []
    labels = []
    predictions = []
    print('Starting cross-validation...')
    for split in range(splits):
        benchmark_split = split
        out = main(split, benchmark_split)
        out = main(split, benchmark_split, True)
        params_list.append(out)
        labels += out[8][0]
        predictions += out[8][1]
    create_plts(params_list, cross_validation, False, split, root, learning_rate, 
                num_epochs,benchmark_crossvalid = True,labels = labels, predictions = predictions)  
    return params_list, labels, predictions

def main(split,benchmark_split, benchmark = False):
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
            acc, true_mcc, loss_ave, cm , mcc_orga, cm_orga, label_predicted_batch = validate(bench_loader, model, dev)
            print('Confusion matrix is:\n', cm)
            mcc_res_post, mcc_glob_post, mcc_glob_pre, cs_pre, cs_post, csreldiff_pre, csreldiff_post = createcompfile(root,label_predicted_batch,benchmark_split, true_mcc)
            create_plts(cm, cross_validation, benchmark, benchmark_split, root, learning_rate, num_epochs, mcc_orga = mcc_orga, cm_orga = cm_orga)
            out_params = [true_mcc, mcc_res_post, mcc_glob_pre, mcc_glob_post, cs_pre, cs_post, csreldiff_pre, csreldiff_post, label_predicted_batch , cm]
            return out_params
        except:
            print('No model found for benchmarking! Start a new run with benchmark = False!')
    else:
        print('Validationset is:', split, 'Benchmarkset is:', benchmark_split)
        # train on the GPU or on the CPU, if a GPU is not available
        train_data, train_labels, validation_data, validation_labels, info, train_orga, validation_orga = de_serializeInput(split, benchmark_split)
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
def de_serializeInput(validation_split, benchmark_split):    
    all_features, info = loaddata('signalP4.npz', 'train_set.fasta')
    train_data,train_labels,train_orga = [],[],[]
    try:       
        print('Loading pickled files...')
        for split in range(splits):
            split_data = pickle.load(open(root+"pickled_files\\split_"+str(split)+"_data.pickle", "rb"))
            split_labels = pickle.load(open(root+"pickled_files\\split_"+str(split)+"_labels.pickle", "rb"))
            split_orga  = pickle.load(open(root+"pickled_files\\split_"+str(split)+"_orga.pickle", "rb"))
            if split == benchmark_split and not benchmarked_cross_validation:
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
            if split == benchmark_split and not benchmarked_cross_validation:
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

def createcompfile(root, label_predicted_batch,split, mcc_pre):
    k,j = 0, 0
    f = open(root+"comparison"+str(split)+".txt","w+")
    mcc_glob_pre = calcGlobMCC(label_predicted_batch)
    csdiff_pre, csreldiff_pre = csdiff(label_predicted_batch)
    gaps, mixed, label_predicted_batch, org_pred = postProcess(label_predicted_batch)
    mcc_glob_post = calcGlobMCC(label_predicted_batch)
    csdiff_post, csreldiff_post = csdiff(label_predicted_batch)
    mcc_post, cm, acc = calcResMCC(label_predicted_batch)
    f.write("Mean residue cleavage residue deviation of predicted to true label before postprocessing: " + str(round(csdiff_pre,3)) + " and after " + str(round(csdiff_post,3)) +
            "\nGlobulal signal peptide MCC before post-processing: " + str(round(mcc_glob_pre,3)) + " and after: " + str(round(mcc_glob_post,3)) +
            "\nResidue MCC before post-processing: " + str(round(mcc_pre,3)) + " and after: " + str(round(mcc_post,3)) +
            "\nGaps have been found at protein predictions: "+ str(gaps) + " and have been post-processed" +
            "\nMixed Signal peptide predictions have been found at: "+ str(mixed) + " and have been post-processed\n")
    for i in range(len(label_predicted_batch[0])):
        f.write("Protein "+ str(i)+ "\n")
        f.write("True labels:      " + str(label_predicted_batch[0][i].tolist()) + "\n")
        if i in gaps: 
            f.write("Orig pred labels: " + str(org_pred[0][j].tolist()) + "\n") 
            j += 1
        if i in mixed:
            f.write("Orig pred labels: " + str(org_pred[1][k].tolist()) + "\n") 
            k += 1
        f.write("Predicted labels: " + str(label_predicted_batch[1][i].astype(int).tolist()) + "\n")        
    f.close()
    return mcc_post, mcc_glob_post, mcc_glob_pre, csdiff_pre, csdiff_post, csreldiff_pre, csreldiff_post

def postProcess(label_predicted_batch):
    gaps, mixed, org_pred = [], [], [[], []]
    gapstr,mixedstr = [],[]
    for i in range (len(label_predicted_batch[0])):
        truth, prediction = label_predicted_batch[0][i], label_predicted_batch[1][i]
        gap, mixedtype, prediction = processPrediction(prediction,org_pred)
        gapsi,mixedsi,_ = processPrediction(truth,[[],[]])
        if gap:
            gaps.append(i)
        if gapsi:
            gapstr.append(i)
        if mixedsi:
            mixedstr.append(i)
        if mixedtype:
            mixed.append(i)
        label_predicted_batch[1][i] = prediction 
    print('The prediction contains gaps at : ' + str(gaps))      
    print('The true labels contain gaps at : ' + str(gapstr))
    print('The prediction contains mixed SP types at : ' + str(mixed)) 
    print('The true labels contain mixed SP types at : ' + str(mixedstr))
    return gaps, mixed, label_predicted_batch, org_pred
    
def processPrediction (prediction, org_pr):
    gap = False
    mixedtype = False
    endnotNull = (prediction[69] != 0)
    x = (prediction == 0)  
    x = np.where(x[:-1] != x[1:])[0]
    if x.size != 0:
        if  x.size > 1 or endnotNull:
            gap = True
            org_pr[0].append(prediction.astype(int))
        if np.unique(prediction).size > 2 and not gap:
            mixedtype = True
            org_pr[1].append(prediction.astype(int)) 
        if endnotNull:
            gapstart = x[0]+1
            if prediction[x[0]] == 0 and x.size > 1 : gapstart = x[1]+1
            prediction[gapstart:] = 0
            x = (prediction == 0)  
            x = np.where(x[:-1] != x[1:])[0]
        if x.size > 1 or np.unique(prediction).size > 2:
            gap_end = x[x.size-1]+1
            most_common_residue = np.bincount(prediction[:gap_end].astype(int)).argmax()
            prediction[:gap_end] = most_common_residue  
    return gap, mixedtype, prediction

def calcGlobMCC(label_predicted_batch):
    x = [label[0] for label in label_predicted_batch[0]]
    y = [label[0] for label in label_predicted_batch[1]]
    mcc = metrics.matthews_corrcoef(x, y)
    return mcc

def csdiff(label_predicted_batch):
    csreldiff = []
    csdiff = 0
    label = label_predicted_batch[0]
    prediction = label_predicted_batch[1]
    for x in range (len(label_predicted_batch[0])):
        csreldiff.append(label[x][label[x] != 0].size - prediction[x][prediction[x] != 0].size)
        csdiff += abs(label[x][label[x] != 0].size - prediction[x][prediction[x] != 0].size)
    csdiff = csdiff/len(label_predicted_batch[0])
    return csdiff, csreldiff   
 
def calcResMCC(label_predicted_batch):
    x = [list(label) for label in label_predicted_batch[0]]
    y = [list(label) for label in label_predicted_batch[1]]
    mcc, cm, acc = calcMCCbatch(x,y)
    return mcc,cm, acc

def calcClassImbalance(info):
# calculate class imbalance of the dataset 
    # for 6 classes: counts = [0,0,0,0,0,0]
    counts = [0,0,0,0]
    for x in info:
        classes = info[x][3]
        counts[0] = counts[0] + classes.count('I')
        counts[0] = counts[0] + classes.count('M')
        counts[0] = counts[0] + classes.count('O')
        counts[1] = counts[1] + classes.count('S')
        counts[2] = counts[2] + classes.count('T')
        counts[3] = counts[3] + classes.count('L')
    counts = [1/x for x in counts]
    return counts
# =============================================================================
# Functions for training/validation
# =============================================================================

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
    x = sum(predicted_batch, [])
    y = sum(labels_batch,[])
    mcc = metrics.matthews_corrcoef(x, y)
    cm = metrics.confusion_matrix(x, y,  [0, 1, 2, 3]) #[0, 1, 2, 3, 4, 5]) 
    acc = metrics.accuracy_score(x,y)
    return mcc,cm,acc

def calcMCCorga(labels_batch, predicted_batch):
#   calculate MCC over given batches of an epoch in training/validation 
#   [0]:Archea, [1]:Eukaryot, [2]:Gram negative, [3]:Gram positive 
    mcc_list, cm_list = [],[]
    for x in range(len(labels_batch)):   
        mcc = metrics.matthews_corrcoef(predicted_batch[x], labels_batch[x])
        cm =  metrics.confusion_matrix(predicted_batch[x], labels_batch[x],  [0, 1, 2, 3]) #[0, 1, 2, 3, 4, 5])  
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
            if no_crf:
                loss = criterion(outputs, labels)
            else: 
                loss = criterion(outputs, labels)
                outputs = outputs.permute(2,0,1) 
                loss = -model.crf(outputs, labels.permute(1,0), mask = mask.permute(1,0))+loss
                
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Track the accuracy, mcc and cm                    
            if (epoch%printafterepoch) == 0: 
                if no_crf:
                    _, predicted = torch.max(outputs.data, 1)                
                    predicted = predicted.squeeze_()
                    correct += (predicted == labels).sum().item()  
                else:
                    predicted = nn.Tensor(model.crf.decode(outputs)).cuda()
                    correct += (predicted == labels.float()).sum().item()  
                total += labels.size(0)* labels.size(1)
                predicted_batch, labels_batch, label_predicted_batch = orgaBatch(labels, predicted, orga, predicted_batch, labels_batch, label_predicted_batch)
                loss_train_list.append(loss.item())  
                
        # and print the results
        if (epoch%printafterepoch) == 0:            
            mcc_train, cm_train, a = calcMCCbatch(labels_batch, predicted_batch)
            acc_train = (correct / total)*100
            loss_ave = sum(loss_train_list)/len(loss_train_list)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, MCC: {:.2f}'
                  .format(epoch+1, num_epochs, i + 1, total_step, loss_ave,
                          acc_train, mcc_train))
            acc_valid, mcc_valid, loss_valid, cm_valid, mcc_orga, cm_orga, label_predicted_batch_val = validate(validation_loader, model, dev)
            out_params.append((loss_valid, loss_ave, epoch, acc_valid, acc_train, mcc_valid, mcc_train , mcc_orga, cm_orga, cm_train, cm_valid))
    # check overfitting
    print('Best validation loss:', min(out_params)[0]  ,' at epoch:', min(out_params)[2])
    return model, out_params, label_predicted_batch

def validate(validation_loader, model, dev):
    with torch.no_grad():   
        model.eval()
        correct = 0
        total = 0
        predicted_batch = [[],[],[],[]] # [0]:Archea, [1]:Eukaryot, [2]:Gram negative, [3]:Gram positive
        labels_batch= [[],[],[],[]]
        label_predicted_batch = [[],[]]
        loss_list = []
        criterion = torch.nn.CrossEntropyLoss(weight = class_weights, ignore_index = -100, reduction = 'mean')
        for validation, labels, mask, orga in validation_loader:
            # preprocess outputs to correct format (1024*70*1)
            validation, labels, mask = validation.to(dev), labels.to(dev), mask.to(dev)
            outputs = model(validation.unsqueeze(3))
            outputs.squeeze_()
            if no_crf:
                # use CrossEntropyloss minimalization
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            else: 
                # apply conditional random field and decode via Vertibri algorithm
                loss = criterion(outputs, labels)
                outputs = outputs.permute(2,0,1)
                loss = -model.crf(outputs, labels.permute(1,0), mask = mask.permute(1,0))+loss
                predicted = nn.Tensor(model.crf.decode(outputs)).cuda()
                correct += (predicted == labels.float()).sum().item()
            # calculate quality measurements
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
    if cross_validation and not normal_run and not gridsearch:
        print("Starting normal cross-validation run...")
        out1,out2, labels, predictions = cross_validate() 
        create_plts(out1, cross_validation, False, selected_split, root, learning_rate, num_epochs)  
        create_plts(out2, cross_validation, False, selected_split, root, learning_rate, num_epochs,benchmark_crossvalid = True, labels =labels, predictions= predictions) 
    else: print('Disable normal run and gridsearch to do simple cross validation!')
    if gridsearch :     
        if cross_validation:
            print ("Disable cross validation to do gridsearch.")
        else:
            cross_valid_params = []
            print("Starting gridsearch... This can take up to a day or two...")
    #        for y in range(len(all_learning_rate)):
    #            learning_rate = all_learning_rate[y]
    #            out = cross_validate()
    #            cross_valid_params.append(out)
            learning_rate = 1e-3
            for y in range(len(all_num_epochs)):            
                num_epochs = all_num_epochs[y]
                out = cross_validate()
                cross_valid_params.append(out)
    if normal_run:
        cross_validation, gridsearch = False, False
        out = main(selected_split, benchmark_split)
    if benchmark:
        out = main(0, benchmark_split, benchmark = True)
    if benchmarked_cross_validation: 
        out, labels, predictions = cross_benchmark()
    print("Runtime: ", time.time() - timer)