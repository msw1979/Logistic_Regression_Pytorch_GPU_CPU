# Author Dr. M. Alwarawrah
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
# PyTorch Library
import torch

# Import Class Linear
from torch.nn import Linear

# Library for this section
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.metrics import (r2_score,roc_auc_score,hinge_loss,confusion_matrix,classification_report,mean_squared_error,jaccard_score,log_loss)

torch.manual_seed(4)

#print if the code is using GPU/CUDA or CPU
if torch.cuda.is_available() == True:
    print('This device is using CUDA')
    device = torch.device("cuda:0")
    os.environ['CUDA_VISIBLE_DEVICES'] ='0'
else:    
    print('This device is using CPU')
    device = torch.device("cpu")

# start recording time
t_initial = time.time()

#normalization function
def normalize(x):
    norm = (x - x.mean()) / x.std()
    return norm

# Create logistic_regression class
class logistic_regression(nn.Module):
    
    # Constructor
    def __init__(self, n_inputs, n_output):
        super(logistic_regression, self).__init__()
        # neural network with n_inputs and 100 neurons
        self.linear1 = nn.Linear(n_inputs, 100)
        #Relu activation
        self.relu = nn.ReLU()
        #neural network with 100 neurons and n_output
        self.output = nn.Linear(100, n_output)
        self.sigmoid = nn.Sigmoid()
 

    # Prediction
    def forward(self, x):
        x = self.relu(self.linear1(x))
        y = self.sigmoid(self.output(x))
        return y

#confusion Matrix
def conf_mat(y_test, yhat, name):
    #calculate confusion matrix
    CM = confusion_matrix(y_test, yhat, labels=[0,1])

    #plot confusion matrix
    plt.clf()
    fig, ax = plt.subplots()
    ax.matshow(CM, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            ax.text(x=j, y=i,s=CM[i, j], va='center', ha='center', size='xx-large')
    plt.xticks(np.arange(0, 2, 1), ['Stay','Leave'])
    plt.yticks(np.arange(0, 2, 1), ['Stay','Leave'])
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix %s'%(name), fontsize=16)
    plt.savefig('confusion_matrix_%s.png'%(name))

#class prediction function for binary system
def predict_class(yhat, thresh=0.5):
  # Return a tensor with  1 if y_pred > 0.5, and 0 otherwise
  y_pred = (yhat > thresh).to(torch.float32)
  return y_pred

#Accuracy function
def acc(y_pred, y):
    # Return the proportion of matches between `y_pred` and `y`, 
    # then average it over the number of samples
    accuracy = (y_pred.cpu()==y.cpu()).sum().item()/len(y.data)
    return accuracy

# Train the model
def train_model(model, train_loader, x_train, y_train, x_test,y_test, optimizer, criterion, epochs, output_file):
    #define list
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(epochs):
        #set initial values for loss and accuracy sum to zero
        loss_sum = 0
        accuracy_sum = 0
        #training
        for x, y in train_loader: 
            #load the data to device
            x,y = x.to(device), y.to(device)
            #get prediction        
            model.train()
            z = model(x)
            #calculate loss        
            loss = criterion(z, y)
            #sum accuracy and loss
            y_pred = predict_class(z)
            accuracy_sum += acc(y_pred,y)
            loss_sum += loss.data.item() 
            #Sets the gradients of all optimized torch.Tensor s to zero        
            optimizer.zero_grad()
            # computes dloss/dx for every parameter x        
            loss.backward()
            #Performs a single optimization step (parameter update)        
            optimizer.step()
        #append values
        train_loss.append(loss_sum)
        train_acc.append(accuracy_sum/len(train_loader))

        #print the training infor to screen and file
        print("Training, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch, loss_sum, accuracy_sum/len(train_loader)) , file=output_file) 
        print("Training, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch, loss_sum, accuracy_sum/len(train_loader))) 

        #validation
        #load the data to device
        x_test, y_test = x_test.to(device), y_test.to(device)
        #get prediction        
        z = model(x_test)
        #calculate loss
        loss = criterion(z, y_test)
        #calculate accuracy
        y_pred = predict_class(z)
        accuracy = acc(y_pred,y_test)
        #append values
        val_acc.append(accuracy)
        val_loss.append(loss.data.item())
        #print the validation infor to screen and file
        print("Validation, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch, loss.data.item(), accuracy) , file=output_file) 
        print("Validation, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch, loss.data.item(), accuracy)) 

    #print confusion matrix
    x_test, y_test = x_test.to(device), y_test.to(device)
    model.eval()
    z = model(x_test)
    yhat = z.data.round().cpu()
    conf_mat(y_test.cpu(), yhat, 'Validation')

    #print confusion matrix
    x_train, y_train = x_train.to(device), y_train.to(device)
    z = model(x_train)
    yhat = z.data.round().cpu()
    conf_mat(y_train.cpu(), yhat, 'Training')

    return train_loss, train_acc, val_loss, val_acc

#plot Loss and Accuracy vs epoch
def plot_loss_accuracy(train_loss, train_acc, val_loss, val_acc):
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(train_loss, color='k', label = 'Training Loss')
    ax.plot(val_loss, color='r', label = 'Validation Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=16)
    ax2 = ax.twinx()
    #ax2.plot(train_acc, color='b', label = 'Training Accuracy')
    ax2.plot(train_acc, color='b', label = 'Training Accuracy')
    ax2.plot(val_acc, color='g', label = 'Validation Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=16)
    fig.legend(loc ="center")
    fig.tight_layout()
    plt.savefig('loss_accuracy_epoch.png')

#Read dataframe
# data from https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv
raw_df = pd.read_csv('ChurnData.csv')

output_file = open('output.txt','w')
print('Torch version: {}'.format(torch.__version__), file=output_file)

# print number of observations and features in the data before cleaning
print("There are " + str(len(raw_df)) + " observations in the dataset.", file = output_file)
print("There are " + str(len(raw_df.columns)) + " variables in the dataset.", file = output_file)

# display first rows in the dataset
print(raw_df.head(), file = output_file)


#plot a histogram of the tip amount
plt.clf()
raw_df.hist()
plt.tight_layout()
plt.savefig('hist.png')

#----- Data preprocess -----#
#select the following features
df = raw_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]

# print number of observations and features in the data before cleaning
print("There are " + str(len(df)) + " observations in the dataset.", file = output_file)
print("There are " + str(len(df.columns)) + " variables in the dataset.", file = output_file)
print("{}".format(df.columns), file = output_file)
#----- Data preprocess End-----#

#----- define Features and target data set-----#
features = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
target = np.asarray(df['churn'].values)

#----- split train/test-----#
X_train, X_test, y_train, y_test = train_test_split( features, target, test_size=0.3, random_state=4)

#normalize features
x_train = normalize(X_train)
x_test = normalize(X_test)

#convert to tensor
x_train = torch.tensor(x_train).to(torch.float32)
x_test = torch.tensor(x_test).to(torch.float32)
y_train = torch.tensor(y_train).to(torch.float32).view(-1, 1)
y_test = torch.tensor(y_test).to(torch.float32).view(-1, 1)


#print train and test set shapes
print ('Train set:', x_train.shape,  y_train.shape, file = output_file)
print ('Test set:', x_test.shape,  y_test.shape, file = output_file)

train_tensor = data_utils.TensorDataset(x_train, y_train) 
#val_tensor = data_utils.TensorDataset(x_test, y_test) 

input_dim = x_train.shape[1]
output_dim = 1

train_loader=DataLoader(dataset=train_tensor, batch_size=50)
#validation_loader = DataLoader(dataset=val_tensor, batch_size=20)

# Sequential Model
#you can also use Sequential
#model = nn.Sequential( nn.Linear(input_dim, 100), nn.ReLU(), nn.Linear(100, output_dim),  nn.Sigmoid())

#Class model
model = logistic_regression(input_dim, output_dim)
#load the model to device
model = model.to(device)

learning_rate = 0.1
epochs = 200

# define criterion to calculate the loss. Binary Cross Entropy Loss
criterion = nn.BCELoss() 
#define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#train model
train_loss, train_acc, val_loss, val_acc = train_model(model, train_loader, x_train, y_train, x_test,y_test, optimizer, criterion, epochs, output_file)

plot_loss_accuracy(train_loss, train_acc, val_loss, val_acc)

output_file.close()

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))