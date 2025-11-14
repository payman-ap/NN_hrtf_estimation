from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from scipy.io import loadmat
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

## 01 Importing NN ################################################
# from LSTM_NETs import LSTM
from lstm_net_tut2 import LSTMnet

# Initialization
N = 36 # batch_size
L = 128 # seq_length
D = 1 # if bidirectional True:2 else if bidirec.. False: 1
Hin = 2 # input_feat_size
Hcell = 256 # hidden_unit_size
Hout = Hcell # =proj_size (whatever size needed)  or Hcell in case proj_size is not defined, usually in this scenario an FC layer will be defined
num_layers = 4 # number of LSTM layers
proj_size = 1

BATCH_SIZE = N
SHUFFLE = False
LEARNING_RATE = 0.0001
EPOCHS = 5000

net = LSTMnet(N,  L, D, Hin, Hcell, Hout, num_layers, proj_size).to(device) # .cuda()

## 02 Log settings ################################################
from torch.utils.tensorboard import SummaryWriter
RunLabel = 'LSTMnet_TEST3_to_eval2dAz'
name_extension = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")      # "%Y_%m_%d_%I_%M_%S_%p"
LOGFOLDER = 'runs_lstmnet/' + name_extension + '_' + RunLabel
writer = SummaryWriter(log_dir=LOGFOLDER)

## Importing Data
from dataprovider import DataHRIR
# Trainloader
# input_path_train = "./DATA/DATA_CIPIC_6/input_train70_lstm_azimuth36.mat"
# hrir_path_train = "./DATA/DATA_CIPIC_6/target_train70_lstm_azimuth36.mat"
input_path_train = "./DATA/DATA_CIPIC_6/input_train70_lstm_azimuth36.mat"
hrir_path_train = "./DATA/DATA_CIPIC_6/target_train70_lstm_azimuth36.mat"
train_dataset = DataHRIR(input_path_train, hrir_path_train, noSubjID=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = False)
# Testloader
# input_path_test = "./DATA/DATA_CIPIC_6/input_test22_lstm_azimuth36.mat"
# hrir_path_test = "./DATA/DATA_CIPIC_6/target_test22_lstm_azimuth36.mat"
input_path_test = "./DATA/DATA_CIPIC_6/input_test22_lstm_azimuth36.mat"
hrir_path_test = "./DATA/DATA_CIPIC_6/target_test22_lstm_azimuth36.mat"
test_dataset = DataHRIR(input_path_test, hrir_path_test, noSubjID=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = False)


## Setting training criterias

loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)  # optim.SGD(net.parameters(), lr=LEARNING_RATE)


def train_single_epoch(net, train_loader, loss_fn, optimizer, device):

    total_loss = 0


    for index, data in tqdm(enumerate(train_loader, 0)):

        input, target = data

        input = Variable(input).to(device).float()
        target = Variable(target).to(device).float()


        # print('01 ', torch.Tensor.size(input))
        # print('02 ', torch.Tensor.size(target))
        # ===================forward=====================
        # calculate loss
        prediction = net(input[:,:,0:Hin])
        loss = loss_fn(prediction, torch.transpose(target[:,0:1,:],1,2))


        # ===================backward====================
        optimizer.zero_grad()    # clear previous batch gradients for new batch train
        loss.backward()          # backpropagation, compute gradients
        optimizer.step()         # apply gradients
        total_loss += loss.item()
    
    return total_loss

def validation(net, test_loader, loss_fn, device):
    val_total_loss = 0
    
    with torch.no_grad():
        for testinput, testtarget in test_loader:

            testinput = testinput.to(device)
            testtarget = testtarget.to(device)

            

            testinput = Variable(testinput).to(device).float()
            # testitd = torch.round(testitd) # *44.1e3 for sample wise ITD using networks with linear output function
            testtarget = Variable(testtarget).to(device).float()


            esttarget = net(testinput[:,:,0:Hin])                                                                                    ######## ======> changed for autoencoder  org: net(inputs) 
            val_loss = loss_fn(esttarget,  torch.transpose(testtarget[:,0:1,:],1,2))
            val_total_loss += val_loss.item()

    return val_total_loss


## Training loop

for epoch in tqdm(range(EPOCHS)):

    total_loss = train_single_epoch(net, train_loader, loss_function, optimizer, device)
    net.eval()
    val_total_loss = validation(net, test_loader, loss_function, device)
    net.train()


    print(f'epoch [{epoch+1}/{EPOCHS}], loss:{total_loss:.8f}')
    writer.add_scalar('training loss', total_loss, epoch)    # epoch * n_total_steps + i
    print(f'epoch [{epoch+1}/{EPOCHS}], valloss:{val_total_loss:.8f}')
    # print('----->>>>>>>>>>>>>>>   ', val_total_loss)
    writer.add_scalar('validation loss', val_total_loss, epoch)


    if (epoch+1) % 20 == 0:
        name_extension2 = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        PATH = './models_lstmnet2/model_' + name_extension2 + '_' + RunLabel + '_epoch_' + str(epoch+1 ) + '.pth'
        torch.save(net, PATH)



print('Finished Training')

######################## SAVING MODEL ########################
# name_extension = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
PATH = './models_lstmnet2/model_' + name_extension + '_' + RunLabel + '_final.pth'

torch.save(net, PATH)
print('Model Saved ...')

print('I AM HERE .....')
a, b = train_dataset.__getitem__(0)

print(torch.Tensor.size(a))
# a = torch.unsqueeze(a,0)
a = torch.randn(BATCH_SIZE,L,Hin)


writer.add_graph(net, a.to(device))    

print('... Training finished successfully!')

