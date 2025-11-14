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

# 01 Importing NN ################################################
# from LSTM_NETs import LSTM
from lstm_net_tut2 import LSTM, LSTMnet

input_size, hidden_size, num_layers, output_dim = 39, 128, 1, 128
net = LSTM(input_size, hidden_size, num_layers, output_dim).to(device) # .cuda()



net.train()


EPOCHS = 50
BATCH_SIZE = 36
SHUFFLE = False
LEARNING_RATE = 0.001

# 02 Log settings ################################################
from torch.utils.tensorboard import SummaryWriter
RunLabel = 'LSTM_1L128O'
name_extension = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")      # "%Y_%m_%d_%I_%M_%S_%p"
LOGFOLDER = 'runsITD/' + name_extension + '_' + RunLabel
writer = SummaryWriter(log_dir=LOGFOLDER)


# Importing Data
from dataprovider import DataHRTF

input_path_train = "./DATA/DATA_CIPIC_6/input_train.mat"
hrir_path_train = "./DATA/DATA_CIPIC_6/target_train.mat"
itd_path_train = "./DATA/DATA_CIPIC_6/ITDs/target_train_itd.mat"

input_path_test = "./DATA/DATA_CIPIC_6/input_test.mat"
hrir_path_test = "./DATA/DATA_CIPIC_6/target_test.mat"
itd_path_test = "./DATA/DATA_CIPIC_6/ITDs/target_test_itd.mat"


trainDataset = DataHRTF(input_path_train, hrir_path_train, itd_path_train, noSubjID=False)
testDataset = DataHRTF(input_path_test, hrir_path_test, itd_path_test, noSubjID=False)

train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle = SHUFFLE)
test_loader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle = SHUFFLE)

# print('***---- ' ,test_loader.__len__())

loss_fn = nn.MSELoss()   # Criterion
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE) # optim.SGD(net.parameters(), lr=LEARNING_RATE)   OR  optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0e-5)

def train_single_epoch(net, train_loader, loss_fn, optimizer, device):

    total_loss = 0


    for index, data in tqdm(enumerate(train_loader, 0)):

        input, target, itd = data

        input = Variable(input).to(device).float()
        target = Variable(target).to(device).float()
        itd = Variable(itd).to(device).float()


        # print('01 ', torch.Tensor.size(input))
        # print('02 ', torch.Tensor.size(target))
        # ===================forward=====================
        # calculate loss
        prediction = net(input[:,:,:])
        loss = loss_fn(prediction, target)


        # ===================backward====================
        optimizer.zero_grad()    # clear previous batch gradients for new batch train
        loss.backward()          # backpropagation, compute gradients
        optimizer.step()         # apply gradients
        total_loss += loss.item()
    
    return total_loss

def validation(net, test_loader, loss_fn, device):
    val_total_loss = 0
    
    with torch.no_grad():
        for testinput, testtarget, testitd in test_loader:

            testinput = testinput.to(device)
            testtarget = testtarget.to(device)
            testitd = testitd.to(device)

            

            testinput = Variable(testinput).to(device).float()
            # testitd = torch.round(testitd) # *44.1e3 for sample wise ITD using networks with linear output function
            testtarget = Variable(testtarget).to(device).float()
            testitd = Variable(testitd).to(device).float()


            esttarget = net(testinput[:,:,:])                                                                                    ######## ======> changed for autoencoder  org: net(inputs) 
            val_loss = loss_fn(esttarget, testtarget)
            val_total_loss += val_loss.item()

    return val_total_loss






for epoch in tqdm(range(EPOCHS)):

    total_loss = train_single_epoch(net, train_loader, loss_fn, optimizer, device)
    net.eval()
    val_total_loss = validation(net, test_loader, loss_fn, device)
    net.train()


    print(f'epoch [{epoch+1}/{EPOCHS}], loss:{total_loss:.8f}')
    writer.add_scalar('training loss', total_loss, epoch)    # epoch * n_total_steps + i
    print(f'epoch [{epoch+1}/{EPOCHS}], valloss:{val_total_loss:.8f}')
    # print('----->>>>>>>>>>>>>>>   ', val_total_loss)
    writer.add_scalar('validation loss', val_total_loss, epoch)


    # if (epoch+1) % 500 == 0:
    #     name_extension2 = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    #     PATH = 'model_' + name_extension2 + '_' + RunLabel + '_epoch_' + str(epoch ) + '.pth'
    #     torch.save(net, PATH)






print('Finished Training')

######################## SAVING MODEL ########################
# name_extension = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
PATH = 'model_' + name_extension + '_' + RunLabel + '_final.pth'

torch.save(net, PATH)
print('Model Saved ...')

print('I AM HERE .....')
a, b, c = trainDataset.__getitem__(0)

print(torch.Tensor.size(a))
# a = torch.unsqueeze(a,0)
a = torch.randn(1,1,39)


writer.add_graph(net, a.to(device))    

print('... Training finished successfully!')

