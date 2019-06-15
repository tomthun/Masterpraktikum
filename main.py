# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:36:46 2019

@author: Thomas
"""
from CustomDataset import CustomDataset
from CNN import SimpleCNN
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import os

params = {'batch_size': 250,
          'shuffle': True,
          'num_workers': 0}
num_epochs = 100
learning_rate = 1e-5
split = '4' 
cross_validation = True

def cross_validate():
    acc_list = [] 
    epoch_list = []
    print('Starting cross-validation...')
    for splits in range(int(split)+1):
        acc, epoch = main(str(splits))
        acc_list.append(acc)
        epoch_list.append(epoch)
        if (epoch/num_epochs) == 1:
            print('No overfitting!')
        else:
            print('Overfitting after epoch:', epoch,'!')
    print('Mean of the accuracy:', (sum(acc_list)/len(acc_list)))     
    return acc_list, epoch_list

def main(split):
    print('Validationset is:', split)
    # train on the GPU or on the CPU, if a GPU is not available
    dev = torch.device('cuda')
    #dev = torch.device('cpu')
    root = 'C:\\Users\\Thomas\\Documents\\Python_Scripts\\MasterPrak_Data\\'
    train_data, train_labels, validation_data, validation_labels = de_serializeInput(root,split)
    train_dataset = CustomDataset(train_data,train_labels)
    validation_dataset = CustomDataset(validation_data,validation_labels)
    train_loader = DataLoader(train_dataset, **params)
    validation_loader = DataLoader(validation_dataset, **params)
    if cross_validation and os.path.isfile(root + 'model.pickle') and split != '0':
        model = torch.load(root + 'model.pickle')
    else:    
        model = SimpleCNN()
        model = model.to(dev)
    model,epoch = train(model, train_loader, validation_loader, num_epochs, learning_rate, dev)
    acc = validate(validation_loader,model,dev)
    torch.save(model, root + 'model.pickle')
    return acc, epoch

def de_serializeInput(root,split):    
    try:       
        print('Loading pickled files...')
        train_data = pickle.load(open(root+"pickled_files\\train"+split+".pickle", "rb"))
        validation_data = pickle.load(open(root+"pickled_files\\validation"+split+".pickle", "rb"))
        train_labels = pickle.load(open(root+"pickled_files\\train_label"+split+".pickle", "rb"))
        validation_labels = pickle.load(open(root+"pickled_files\\validation_label"+split+".pickle", "rb"))
        print('Done!')
    except (OSError, IOError):     
        print('Pickled files not found!\nCreating new train/validation dataset...')
        all_features, info = loaddata(root,'signalP4.npz', 'train_set.fasta')
        train_keys, validation_keys = selectTestTrainSplit(info,split)
        train_data, train_labels = createDataVectors(info,all_features,train_keys)
        validation_data, validation_labels = createDataVectors(info,all_features, validation_keys)
        pickle.dump(train_data, open( root+"pickled_files\\train"+split+".pickle", "wb" ))
        pickle.dump(validation_data, open( root+"pickled_files\\validation"+split+".pickle", "wb" ))
        pickle.dump(train_labels, open( root+"pickled_files\\train_label"+split+".pickle", "wb" ))
        pickle.dump(validation_labels, open( root+"pickled_files\\validation_label"+split+".pickle", "wb" ))
        print('Saved and Done!')
    return train_data,train_labels,validation_data,validation_labels
       
def loaddata (root, data_name , training_name):
    train_data = open(root+training_name, 'r') 
    train_data = train_data.read().split('\n')
    tmp = np.load(root+data_name)
    info = {}
    header = train_data[0].split('|')[0].replace('>','')
    signalp = train_data[0].split('|')[2]
    partition = train_data[0].split('|')[3]
    seq = train_data[1]
    sig = train_data[2]
    sigbin = list(map(int,sig.replace('I','0').replace('M','1').replace('O','2')
                .replace('S','3').replace('T','4').replace('L','5')))
    count = 3
    for x in range(int((len(train_data)-4)/3)):
            if (len(seq) == 70):
                info[header] = [signalp, partition,seq,sig,sigbin]
            seq = train_data[count+1]
            sig = train_data[count+2]
            sigbin = list(map(int,sig.replace('I','0').replace('M','1').replace('O','2')
                .replace('S','3').replace('T','4').replace('L','5')))
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
        label.append(info[key][4])
        data.append(all_features[key][:70])
    return data,label

def selectTestTrainSplit(train_data,x):
    split = [key  for (key, value) in train_data.items() if value[1] == x]
    rest = set(list(train_data.keys()))-set(split)
    return list(rest),split

def calc_roc(test_pred, test_labels, predCutoff = 0.4):
  tp = 0
  fp = 0
  tn = 0
  fn = 0
  for i, pred in enumerate(test_pred):
    if pred.item() > predCutoff and test_labels[i][0] == 1:
      tp = tp + 1
    elif pred.item() > predCutoff:
      fp = fp + 1
    elif test_labels[i][0] == 1:
      fn = fn + 1
    else:
      tn = tn + 1
  return tp, fp, tn, fn

def train(model, train_loader, validation_loader, num_epochs, learning_rate, dev):
    print('Starting to learn...')
    loss_list = []
    total_step = len(train_loader)
    acc_list = []
    criterion = torch.nn.CrossEntropyLoss(reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (train, labels) in enumerate(train_loader):
            # Run the forward pass
            train, labels = train.to(dev), labels.to(dev)
            outputs = model(train.unsqueeze(3))
            outputs, labels = outputs.squeeze_(), labels.squeeze_()
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
    
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Track the accuracy
            total = labels.size(0)* labels.size(1)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted.squeeze_() == labels).sum().item()
            
            # and print the results
        if (epoch%5) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total)*100))
            acc = validate(validation_loader, model, dev)
            acc_list.append((acc,num_epochs))
    
    # check overfitting
    print('Best accurarcy:', max(acc_list)[0]  ,' at epoch:', max(acc_list)[1])
    return model, max(acc_list)[1]
    
def validate(validation_loader, model, dev):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for validation, labels in validation_loader:
            validation, labels = validation.to(dev), labels.to(dev)
            outputs = model(validation.unsqueeze(3))
            _, predicted = torch.max(outputs.data, 1)
            labels, predicted = predicted.squeeze_(), labels.squeeze_()       
            correct += (predicted == labels).sum().item()
            total = total + (labels.size(0) * labels.size(1))
            result = ((correct / total) * 100)
        print('Test Accuracy of the model on the validation proteins is: {} %'.format(result))
    return result

if __name__ == "__main__":
    if cross_validation == True:
        acc_list, epoch_list = cross_validate()
    else:
        main(split)
        