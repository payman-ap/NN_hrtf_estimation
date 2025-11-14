import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fully connected neural network with one hidden layer
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        self.fc = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):
        
        print('000: ', x.size(0))

        print('00 ', torch.Tensor.size(x))
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 

        print('00 h0 ', torch.Tensor.size(h0))
        print('00 c0 ', torch.Tensor.size(c0))

        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
        
        print('00 out: ', torch.Tensor.size(out))

        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        # out = out[:, -1, :]
        # out: (n, 128)
         
        # out = self.fc(out)
        # out: (n, 10)
        return out

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size, hidden_size, num_layers, output_dim = 39, 128, 1, 128
    net = LSTM(input_size, hidden_size, num_layers, output_dim).to(device) # .cuda()

    a = torch.randn(32,1,39)

    y = net(a)

    print('main func size .. ', y.size())





    from torchinfo import summary
    summary(net, (1000,1,39))


    # from torch.utils.tensorboard import SummaryWriter
    # from datetime import datetime
    # name_extension = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # LOGFOLDER = 'Graphs/' + name_extension
    # writer = SummaryWriter(log_dir=LOGFOLDER)

    # a = torch.randn(1,1,39)


    # writer.add_graph(net, a)    

    # # writer.add_graph(net, torch.unsqueeze(a[:,0:5].to(device).float(), 0))    






