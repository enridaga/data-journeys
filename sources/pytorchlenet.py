
import numpy as np 
import pandas as pd 
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split 
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
import csv
import time
import cv2

traindata=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
testdata=pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

trainset=traindata.drop('label',axis=1)
trainlabel=traindata['label']

testset=testdata.drop('id',axis=1)

train_data,validation_data,train_label,validation_label=train_test_split(trainset,trainlabel,test_size=0.01,stratify=trainlabel,random_state=42)

train_data=train_data.to_numpy().astype(np.float32).reshape(len(train_data),28,28,1)
train_label=torch.from_numpy(train_label.to_numpy())
validation_data=validation_data.to_numpy().astype(np.float32).reshape(len(validation_data),28,28,1)
validation_label=torch.from_numpy(validation_label.to_numpy())
test_data=testset.to_numpy().reshape(len(testset),28,28,1)

new_train_data=train_data[0][np.newaxis,:]
new_train_label=np.array([train_label[0]])

def imgtrans(im,datagroup,labelgroup,imglabel):
    global j
    img1=cv2.warpAffine(im,rot_mat1,(28,28),flags=cv2.INTER_LINEAR)[np.newaxis,:,:,np.newaxis]
    img3=cv2.warpAffine(im,scale_mat1,(28,28),flags=cv2.INTER_LINEAR)[np.newaxis,:,:,np.newaxis]
    img5=cv2.warpAffine(im,shift_mat1,(28,28),flags=cv2.INTER_LINEAR)[np.newaxis,:,:,np.newaxis]
    img2=cv2.warpAffine(im,rot_mat2,(28,28),flags=cv2.INTER_LINEAR)[np.newaxis,:,:,np.newaxis]
    img4=cv2.warpAffine(im,scale_mat2,(28,28),flags=cv2.INTER_LINEAR)[np.newaxis,:,:,np.newaxis]
    img6=cv2.warpAffine(im,shift_mat2,(28,28),flags=cv2.INTER_LINEAR)[np.newaxis,:,:,np.newaxis]
    im=im[np.newaxis,:,:,np.newaxis]
    #data=np.concatenate((datagroup[j],im,img1),axis=0)
    #label=np.concatenate((labelgroup[j],np.array([imglabel]),np.array([imglabel])),axis=0)
    data=np.concatenate((datagroup[j],im,img1,img2,img3,img4,img5,img6),axis=0)
    label=np.concatenate((labelgroup[j],np.array([imglabel]),np.array([imglabel]),np.array([imglabel]),np.array([imglabel]),np.array([imglabel]),np.array([imglabel]),np.array([imglabel])),axis=0)
    datagroup[j]=data
    labelgroup[j]=label
    
#start=time.time()
rot_mat1=cv2.getRotationMatrix2D((14,14),10,1)     #旋转
rot_mat2=cv2.getRotationMatrix2D((14,14),-10,1)
scale_mat1=cv2.getRotationMatrix2D((14,14),0,1.1)  #缩放
scale_mat2=cv2.getRotationMatrix2D((14,14),0,0.9)
shift_mat1= np.float32([[1,0,3],[0,1,3]])                                  #平移
shift_mat2 = np.float32([[1,0,-3],[0,1,-3]])
j=0
datadict={}
for i in range(13):
    datadict[i]=new_train_data
labeldict={}
for i in range(13):
    labeldict[i]=new_train_label
start=time.time()
for i in range(len(train_data)):
    img=train_data[i,:,:,0]
    imgtrans(img,datadict,labeldict,train_label[i])
    if i%5000==0:
        print(datadict[j].shape,labeldict[j].shape,time.time()-start)
        start=time.time()
        j+=1
        
for i in range(13):
    new_train_data=np.concatenate((new_train_data,datadict[i]),axis=0)
    new_train_label=np.concatenate((new_train_label,labeldict[i]),axis=0)
    
new_train_data=new_train_data[1:,:,:,:]
new_train_label=new_train_label[1:]

train_data=new_train_data
train_label=torch.from_numpy(new_train_label)

new_train_data=np.concatenate((train_data,test_data),axis=0)
print(new_train_data.shape)

#fig,axes=plt.subplots(10,10,figsize=(10,10))
#for i in range(10):
#    for j in range(10):
#        axes[i][j].axis('off')
#        axes[i][j].imshow(train_data[i*10+j,:,:,0])
#fig.suptitle('train_data')

#fig2,axes2=plt.subplots(10,10,figsize=(10,10))
#for i in range(10):
#    for j in range(10):
#        axes2[i][j].axis('off')
#        axes2[i][j].imshow(test_data[i*10+j,:,:,0])
#fig2.suptitle('test_data')

class mydataset(torch.utils.data.Dataset):
    def __init__(self,data,label=None):
        self.data=data
        self.label=label
        
    def __getitem__(self,index):
        img=self.data[index]
        img=transforms.ToTensor()(img)
        if self.label is not None:
            label=self.label[index]
            return img,label            #train.val
        else:
            return img                  #test
    
    def __len__(self):
        return len(self.data)
trainset=mydataset(train_data,train_label)
train_loader=torch.utils.data.DataLoader(trainset,batch_size=1024,shuffle=True)
valset=mydataset(validation_data,validation_label)
val_loader=torch.utils.data.DataLoader(valset,batch_size=64,shuffle=False)
testset=mydataset(test_data)
test_loader=torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False)
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
class maxout(nn.Module):
            def __init__(self):
                super(maxout,self).__init__()
                
            def forward(self,input):
                x1,x2=torch.chunk(input,2,1)
                output=torch.nn.functional.relu(x2-x1)+x1
                return output
            
class lenet(nn.Module):
        def __init__(self):
            super(lenet,self).__init__()
            self.features=nn.Sequential(
                nn.Conv2d(1,64,3),         #26
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.Conv2d(64,64,3),        #24
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
                nn.Conv2d(64,128,3),        #22
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                nn.Conv2d(128,128,3),        #20
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2,2),         #10
                nn.Dropout(p=0.2),
                nn.Conv2d(128,256,3),        #8
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.Conv2d(256,256,3),        #6
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2,2),         #3
                nn.Dropout(p=0.2),
            )
            
            self.classify=nn.Sequential(
                nn.Linear(2304,512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Linear(512,10),
            )
        def forward(self,input):
            x=self.features(input)
            x=x.view(x.size(0),-1)
            x=self.classify(x)
            return x
Lenet=lenet()
Lenet.to('cuda')
print('building model......')
print('train......')
epoch=25
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(Lenet.parameters(),lr=0.001,momentum=0.99,weight_decay=0.001)
#lr_scheduler=ReduceLROnPlateau(optimizer,factor=0.5,patience=4)
lr_scheduler=StepLR(optimizer,step_size=5,gamma=0.5)

#train
start=time.time()
for i in range(epoch):
    Lenet.train()
    loss_total=0
    
    for idx,(img,label) in enumerate(train_loader):
            img=img.to('cuda')
            label=label.to('cuda')
            output=Lenet(img)
            loss=criterion(output,label)
            loss_total+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx%100==0:
                    print('epoch:{} , {}/{} , loss:{:.6f}  {:.6f} , lr={}'.format(i,idx,len(train_loader),loss_total/(idx+1),loss.item(),optimizer.param_groups[0]['lr']))
    lr_scheduler.step()
print('time: {} s'.format(time.time()-start))
#validation
print('validate......')
with torch.no_grad():
    Lenet.eval()
    correct=0
    total=600
    for idx,(img,label) in enumerate(val_loader):
            img=img.to('cuda')
            label=label.to('cuda')
            output=Lenet(img)
            prediction=torch.max(output,dim=1)[1]
            correct+=sum(prediction==label).item()
    print('val accuracy:{:.6f}'.format(correct/total))
#pseudo label
print('pseudo label data......')
new_train_label=train_label.to('cuda')

with torch.no_grad():
    Lenet.eval()
    for idx,img in enumerate(test_loader):
        img=img.to('cuda')
        img=img.float()
        output=Lenet(img)
        _,predictions=torch.max(output,1)
        new_train_label=torch.cat((new_train_label,predictions),dim=0)
print(new_train_data.shape)
print(new_train_label.shape)
#retrain
print('retrain with pseudo label......')
new_trainset=mydataset(new_train_data,new_train_label)
new_train_loader=torch.utils.data.DataLoader(new_trainset,batch_size=1024,shuffle=True)

epoch=30
optimizer=torch.optim.Adam(Lenet.parameters(),lr=0.001,weight_decay=0.001)
#optimizer=torch.optim.SGD(Lenet.parameters(),lr=0.001,momentum=0.9,weight_decay=0.001)
lr_scheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=2)
#lr_scheduler=StepLR(optimizer,step_size=5,gamma=0.5)

#train
bestnet=None
bestloss=1
start=time.time()
for i in range(epoch):
    Lenet.train()
    loss_total=0
    
    for idx,(img,label) in enumerate(new_train_loader):
            img=img.to('cuda')
            label=label.to('cuda')
            img=img.float()
            output=Lenet(img)
            loss=criterion(output,label)
            loss_total+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if idx%100==0:
                    print('epoch:{} , {}/{} , loss:{:.6f}  {:.6f} , lr={}'.format(i,idx,len(train_loader),loss_total/(idx+1),loss.item(),optimizer.param_groups[0]['lr']))
            if loss.item()<bestloss:
                bestloss=loss.item()
                bestnet=Lenet
    lr_scheduler.step(loss_total/len(new_train_loader))
print('time: {} s'.format(time.time()-start))
#validation
print('validate......')
with torch.no_grad():
    Lenet.eval()
    correct=0
    total=600
    for idx,(img,label) in enumerate(val_loader):
            img=img.to('cuda')
            label=label.to('cuda')
            output=Lenet(img)
            prediction=torch.max(output,dim=1)[1]
            correct+=sum(prediction==label).item()
    print('val accuracy:{:.6f}'.format(correct/total))
#validation
print('bestnet validate......')
with torch.no_grad():
    bestnet.eval()
    correct=0
    total=600
    for idx,(img,label) in enumerate(val_loader):
            img=img.to('cuda')
            label=label.to('cuda')
            output=bestnet(img)
            prediction=torch.max(output,dim=1)[1]
            correct+=sum(prediction==label).item()
    print('val accuracy:{:.6f}'.format(correct/total))
#test 
print('test data......')
submission=open('submission.csv','w',newline='')
submission=csv.writer(submission)
submission.writerow(['id','label'])

with torch.no_grad():
    bestnet.eval()
    for idx,img in enumerate(test_loader):
        img=img.to('cuda')
        img=img.float()
        output=bestnet(img)
        _,predictions=torch.max(output,1)
        for i in range(len(img)):
            submission.writerow([idx*64+i,predictions[i].item()])
print('finished......')