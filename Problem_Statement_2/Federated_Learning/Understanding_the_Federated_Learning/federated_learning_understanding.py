import os
import logging
import sys
import numpy as np
import pandas as pd
import time
import csv
import random
import matplotlib.pyplot as plt
from PIL import Image
from barbar import Bar
import datetime
import time
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

use_gpu = torch.cuda.is_available()
print(use_gpu)

def log(path, file):
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = " %(message)s"
    file_logging_format = "%(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)
    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

class Config(object):
    def __init__(self):
        self.name = 'fed-chex_res18_v3'
        self.dataset_name = 'Chexpert'
        self.save_path = './ckpt/' + self.name
        self.train_csv = '/workspace/DATASETS/CheXpert-v1.0-small/chexpert-train.csv'
        self.valid_csv = '/workspace/DATASETS/CheXpert-v1.0-small/chexpert-valid.csv'
        self.test_csv =  '/workspace/DATASETS/CheXpert-v1.0-small/chexpert-test.csv'
        
        self.model_name = 'resnet18'  #choose one from resnet18,resnet34
        self.pre_train = True 

        self.num_workers = 8
        self.random_seed=24
        self.img_size = 224

        self.lr = 0.0001      
        self.batch_size = 32
        self.test_batch_size = 1
        self.num_classes = 14
        self.gpu = 0
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
                
        self.com_round = 10
        self.fraction = 1.0
        self.client_epoch = 3
        self.num_clients = 5
        
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = log(path=self.save_path, file=f"{self.name}.logs")
        
opt = Config()
opt.logger.info('Training data Info:')
opt.logger.info(opt.__dict__)

class CheXpertDataSet(Dataset):
    def __init__(self,df,class_names, transform, policy="ones"):
        self.image_filepaths = df["Path"].values
        self.pathologies = class_names
        self.pathologies = sorted(self.pathologies)
        self.csv = df
        
        self.transform = transform
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                mask = self.csv[pathology]
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        if policy == "ones":
            self.labels[self.labels == -1] = 1
        elif policy == "zeroes":
            self.labels[self.labels == -1]= 0
        else:
            self.labels[self.labels == -1] = np.nan
            
    def __getitem__(self, idx):
           
        img = self.image_filepaths[idx]
        image = Image.open(img).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image,label

    def __len__(self):
        return len(self.image_filepaths)
        

def get_trasnformations():
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
    IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)
    
    # Tranform data
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    
    transformList = []
    transformList.append(transforms.Resize((opt.img_size, opt.img_size)))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transformSequence = transforms.Compose(transformList)
    
    return transformSequence
    

def get_datasets():
    train_df = pd.read_csv(opt.train_csv)
    valid_df = pd.read_csv(opt.valid_csv)
    test_df = pd.read_csv(opt.test_csv)
        
    train_df = train_df.fillna(-1)
    valid_df = valid_df.fillna(-1)
    test_df  = test_df.fillna(-1)
    
    # Class names
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                   'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    
    transformations = get_trasnformations()
    
    datasetTrain = CheXpertDataSet(train_df, class_names, transform=transformations, policy = "zeroes")
    datasetValid = CheXpertDataSet(valid_df, class_names, transform=transformations, policy = "zeroes")
    datasetTest  = CheXpertDataSet(test_df,   class_names, transform=transformations, policy = "ones")
    
    opt.logger.info("Train data length: {}".format(len(datasetTrain)))
    opt.logger.info("Valid data length: {}".format(len(datasetValid)))
    opt.logger.info("Test data length: {}".format(len(datasetTest)))
    
#     opt.logger.info(class_names)
    
    return datasetTrain,datasetValid,datasetTest
    

def get_dataloaders():
    datasetTrain,datasetValid,datasetTest = get_datasets()
    
    #for non-federated setting
    # train_loader = DataLoader(datasetTrain,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,pin_memory=True)
    # val_loader = DataLoader(datasetValid,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,pin_memory=True)
    # test_loader = DataLoader(datasetTest,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers,pin_memory=True)
    
    '''
    Divide datasetTrain_ex
    datasetTrain_1, datasetTrain_2, datasetTrain_3, datasetTrain_4, datasetTrain_5, dataleft = random_split(datasetTrain, 
                                                                                                        [100, 100, 100, 100, 100,
                                                                                                         len(datasetTrain) - 500])
    '''
    
    # Divide datasetTrain for number of clients defined in config
    datasetTrain_1, datasetTrain_2, datasetTrain_3, datasetTrain_4, datasetTrain_5 = random_split(datasetTrain, 
                                                                                              [33930,33930,33930,33930,33930])
    
    # Define 5 DataLoaders for each client
    dataLoaderTrain_1 = DataLoader(dataset = datasetTrain_1, batch_size = opt.batch_size,
                                   shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    dataLoaderTrain_2 = DataLoader(dataset = datasetTrain_2, batch_size = opt.batch_size,
                                   shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    dataLoaderTrain_3 = DataLoader(dataset = datasetTrain_3, batch_size = opt.batch_size,
                                   shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    dataLoaderTrain_4 = DataLoader(dataset = datasetTrain_4, batch_size = opt.batch_size,
                                   shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    dataLoaderTrain_5 = DataLoader(dataset = datasetTrain_5, batch_size = opt.batch_size,
                               shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # Define Valid and Test DataLoaders
    dataLoaderVal = DataLoader(dataset = datasetValid, batch_size = opt.batch_size, 
                           shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    dataLoaderTest = DataLoader(dataset = datasetTest, batch_size = 1 ,
                                num_workers = opt.num_workers, pin_memory = True)
    
    dT  = [datasetTrain_1, datasetTrain_2, datasetTrain_3, datasetTrain_4, datasetTrain_5]
    dLT = [dataLoaderTrain_1, dataLoaderTrain_2, dataLoaderTrain_3, dataLoaderTrain_4, dataLoaderTrain_5]
    
    return dT, dLT, dataLoaderVal, dataLoaderTest

class classifier(nn.Module):
    def __init__(self,pretrained=False, out_size=14):
        super(classifier, self).__init__()
        
        if opt.model_name == 'resnet18':
            if pretrained==True:
                self.model = torchvision.models.resnet18(pretrained = True)
            elif pretrained==False:
                self.model = torchvision.models.resnet18(pretrained = False)
                
            num_ftrs = self.model.fc.in_features   
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, out_size),
                                        nn.Sigmoid()
                                        )
            
        if opt.model_name == 'resnet34':
            if pretrained==True:
                self.model = torchvision.models.resnet34(pretrained = True)
            elif pretrained==False:
                self.model = torchvision.models.resnet34(pretrained = False)

            num_ftrs = self.model.fc.in_features   
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, out_size),
                                        nn.Sigmoid()
                                        )

    def forward(self, x):
        x = self.model(x)
        return x

def computeAUROC(dataGT, dataPRED):
    # Computes area under ROC curve 
    # dataGT: ground truth data
    # dataPRED: predicted data
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(opt.num_classes):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
    return outAUROC

def epochTrain(model, dataLoaderTrain, optimizer, loss):
    losstrain = 0
    model.train()

    for batchID, (varInput, target) in enumerate(Bar(dataLoaderTrain)):
        varTarget = target.cuda(non_blocking = True)
        varInput  = varInput.cuda(non_blocking = True)
        varOutput = model(varInput)
        lossvalue = loss(varOutput, varTarget)

        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()

        losstrain += lossvalue.item()

    return losstrain / len(dataLoaderTrain)

def epochVal(model, dataLoaderVal, loss):
    model.eval()
    lossVal = 0
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()

    with torch.no_grad():
        for i, (varInput, target) in enumerate(Bar(dataLoaderVal)):
            
            target = target.cuda(non_blocking = True)
            varInput = varInput.cuda(non_blocking = True)
            
            outGT = torch.cat((outGT, target),0)
            outGT = outGT.cuda(non_blocking = True)
            
            varOutput = model(varInput)
            outPRED = torch.cat((outPRED,varOutput), 0)
            lossVal += loss(varOutput, target)
        aurocIndividual = computeAUROC(outGT, outPRED)
        aurocMean = np.array(aurocIndividual).mean()
#         print('AUROC mean ', aurocMean)

    return lossVal / len(dataLoaderVal),aurocIndividual,aurocMean

def train(client,model, dataLoaderTrain, dataLoaderVal, trMaxEpoch,loss):
    
    optimizer = optim.Adam(model.parameters(), lr =opt.lr, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.0005) 

    # Train the network
    lossMIN = 100000
    best_auc = 0.0
    train_start = []
    train_end = []
    for epochID in range(0, trMaxEpoch):
        train_start.append(time.time())                      # training starts
        losst = epochTrain(model, dataLoaderTrain, optimizer, loss)
        train_end.append(time.time())                        # training ends
        
        lossv,aurocIndividual,aurocMean = epochVal(model, dataLoaderVal, loss)
        
        
        
        opt.logger.info("Train_loss: {:.3f}".format(losst)+'\t'+"Val_loss: {:.3f}".format(lossv)+'\t'+"Val_auc: {:.3f}".format(aurocMean))
        
        with open(f'{opt.save_path}/client_{client}_logs.txt', 'a') as file:
            file.write(str(aurocMean)+','+str(lossv.item())+'\n')
        
        
        if aurocMean> best_auc:
            best_auc = aurocMean
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                        'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
                        os.path.join(opt.save_path,f'client_{client}_{opt.model_name}' + '.pth'))
            
            opt.logger.info('Epoch ' + str(epochID + 1) + ' [save] auc = ' + str(aurocMean))
        else:
            opt.logger.info('Epoch ' + str(epochID + 1) + ' [----] auc = ' + str(aurocMean))

    train_time = np.array(train_end) - np.array(train_start)
    opt.logger.info("Training time for each epoch: {} seconds".format(train_time.round(0)))
    params = model.state_dict()
    return params

def main():
    sel_clients = sorted(random.sample(range(opt.num_clients),round(opt.num_clients*opt.fraction))) # Step 1: select random fraction of clients
    opt.logger.info("The number of clients: {}".format(len(sel_clients)))
    model = classifier(pretrained=opt.pre_train,out_size=opt.num_classes)
    client_models = [classifier(pretrained=opt.pre_train,out_size=opt.num_classes).to(opt.device) for i in range(len(sel_clients))]
    
    loss = torch.nn.BCELoss() 
    global_checkpoint = os.path.join(f'{opt.save_path}',f'global_{opt.name}.pth')
    
    if global_checkpoint != None and os.path.isfile(global_checkpoint):
        modelCheckpoint = torch.load(opt.global_checkpoint,map_location = opt.device)
        model.load_state_dict(modelCheckpoint['state_dict'])
        opt.logger.info("Loaded pre-trained model with success.")
        r_counter=modelCheckpoint['round']
        global_lossMIN = modelCheckpoint['best_loss']
        opt.logger.info('Previously Trained for {} rounds'.format(r_counter))
    else:
        opt.logger.info("Pre-trained weights not found. Training from scratch.")
        e_counter=0
        global_lossMIN = 100000
        best_auc = 0.0
    
    dT, dLT, dataLoaderVal, dataLoaderTesT = get_dataloaders()    
    for r_counter in range(opt.com_round):
        opt.logger.info("[[[ Round {} Start ]]]".format(r_counter + 1))
        params = [None] * opt.num_clients
        
        for i in sel_clients:                                                            # Step 2: send weights to clients
            opt.logger.info("<< Client {} Training Start >>".format(i + 1))
            train_valid_start = time.time()
            params[i] = train(i,client_models[i], dLT[i], dataLoaderVal,                              # Step 3: Perform local computations
                            trMaxEpoch = opt.client_epoch,loss=loss)
            
            train_valid_end = time.time()
            client_time = round(train_valid_end - train_valid_start)
            opt.logger.info("<< Client {} Training End: {} seconds elapsed >>".format(i + 1, client_time))

        fidx = [idx for idx in range(len(params)) if params[idx] != None][0]
        lidx = [idx for idx in range(len(params)) if params[idx] != None][-1]

        for key in params[fidx]:                                                      # Step 4: return updates to server
            weights, weightn = [], []
            for k in sel_clients:
                weights.append(params[k][key]*len(dT[k]))
                weightn.append(len(dT[k]))
            params[lidx][key] = sum(weights) / sum(weightn)                           # weighted averaging model weights
    
    
        #########                                                                    # loading each client model with aggregated weights
        for cl in client_models:
            cl.load_state_dict(params[lidx])
        #########    
        
        model = classifier(pretrained=opt.pre_train,out_size=opt.num_classes)
        model.load_state_dict(params[lidx])                # Step 5: server updates global state
        model.to(opt.device)
        opt.logger.info("[[[ Round {} End ]]]".format(r_counter + 1))
        opt.logger.info('Validating global aggregated model')
        valid_loss,aurocIndividual,aurocMean = epochVal(model, dataLoaderVal,loss)
        
        with open(f'{opt.save_path}/global_logs.txt', 'a') as file:
            file.write(str(r_counter)+','+str(aurocMean)+','+str(valid_loss.item())+'\n')
        
        if aurocMean> best_auc:
            opt.logger.info('auc increased ({:.3f} --> {:.3f}). Saving model ...'.format(best_auc,aurocMean))
            best_auc = aurocMean
            torch.save({'round': r_counter + 1, 'state_dict': model.state_dict(), 
                        'best_loss': global_lossMIN}, 
                        os.path.join(opt.save_path,f'global_{opt.model_name}.pth')) 
            opt.logger.info('Round ' + str(r_counter + 1) + '[save] loss = ' + str(valid_loss.item()) + 'auc = ' + str(aurocMean))
        else:
            opt.logger.info('Round ' + str(r_counter + 1) + ' [----] loss = ' + str(valid_loss.item()) + 'auc = ' + str(aurocMean))
        
    opt.logger.info("Global model trained")

if __name__ == '__main__':
    main()