import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

class DataITD(Dataset):
    def __init__(self, input_path, itd_path, noSubjID = True):

        self.inputtmp = loadmat(input_path)
        self.input = torch.tensor(self.inputtmp["InputData"])
        # self.input = torch.swapaxes(torch.tensor(self.inputtmp["InputData"]), 1, 2)

        # if noSubjID:
        #         self.input = self.input[:,0:2,:]
        print(" in MyDataset init --->>> ", torch.Tensor.size(self.input))

        self.itdtmp = loadmat(itd_path)
        self.itd = torch.tensor(self.itdtmp["TargetITD"])

        self.target = torch.unsqueeze(self.itd, 1)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        x  = self.input[index]
        y = self.target[index]

        return x, y


class DataHRIR(Dataset):
    def __init__(self, input_path, hrir_path, noSubjID = True):

        self.inputtmp = loadmat(input_path)
        self.input = torch.tensor(self.inputtmp["InputData"])
        

        # if noSubjID:
        #         self.input = self.input[:,0:2,:]
        print(" in MyDataset init --->>> ", torch.Tensor.size(self.input))

        self.hrirtmp = loadmat(hrir_path)
        self.hrir = torch.tensor(self.hrirtmp["TargetData"])

        a = torch.unsqueeze(self.hrir[:,0,:],1)
        b = torch.unsqueeze(self.hrir[:,1,:],1)

        self.target = torch.cat( (a,b), 1)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        x  = self.input[index]
        y = self.target[index]

        return x, y



class DataHRTF(Dataset):
    '''
    makes dataloader including Subject Inputs, HRIRs, ITDs;
    gets input_path, hrir_path, itd_path file directories of *.mat file and returns the iterable tuple.
    '''
    def __init__(self, input_path, hrir_path, itd_path, noSubjID = True):

        self.inputtmp = loadmat(input_path)
        self.input = torch.tensor(self.inputtmp["InputData"])
        

        # if noSubjID:
        #         self.input = self.input[:,0:2,:]

        self.hrirtmp = loadmat(hrir_path)
        self.hrir = torch.tensor(self.hrirtmp["TargetData"])

        a = torch.unsqueeze(self.hrir[:,0,:],1)
        b = torch.unsqueeze(self.hrir[:,1,:],1)

        # self.target = torch.cat( (a,b), 1)
        self.target = a

        self.itdtmp = loadmat(itd_path)
        self.itd = torch.tensor(self.itdtmp["TargetITD"])

        self.target_itd = torch.unsqueeze(self.itd, 1)

        print(" in DataHRTF dataloader, input size: --->>> ", torch.Tensor.size(self.input))
        print(" in DataHRTF dataloader, target size: --->>> ", torch.Tensor.size(self.target))
        print(" in DataHRTF dataloader, itd size: --->>> ", torch.Tensor.size(self.target_itd))


    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        Input  = self.input[index]
        hrir = self.target[index]
        itd = self.target_itd[index]


        return Input, hrir, itd





if __name__ == '__main__':

    from tqdm import tqdm

    input_path_train = "./DATA/DATA_CIPIC_6/input_train.mat"
    hrir_path_train = "./DATA/DATA_CIPIC_6/target_train.mat"
    itd_path_train = "./DATA/DATA_CIPIC_6/ITDs/target_train_itd.mat"

    
    train_hrir_dataset = DataHRIR(input_path_train, hrir_path_train, noSubjID=False)
    train_itd_dataset = DataITD(input_path_train, itd_path_train, noSubjID=False)


    train_hrir_loader = DataLoader(train_hrir_dataset, batch_size=32, shuffle = False)
    train_itd_loader = DataLoader(train_itd_dataset, batch_size=32, shuffle = False)

    a1, b1 = train_hrir_dataset.__getitem__(284)
    a2, b2 = train_itd_dataset.__getitem__(284)



    train_hrtf_dataset = DataHRTF(input_path_train, hrir_path_train, itd_path_train, noSubjID=False)
    a3, b3, c3 = train_hrtf_dataset.__getitem__(284)


    for index, data in tqdm(enumerate(train_hrtf_dataset, 0)):

        a, b, c = data

    dummy = 1


    print(torch.Tensor.size(a1), "****" , torch.Tensor.size(b1))
    # print(a1, "++++" , b1)

    print(torch.Tensor.size(a2), "****" , torch.Tensor.size(b2))
    # print(a2, "++++" , b2)

    print(torch.Tensor.size(a3), "****" , torch.Tensor.size(b3), '++++', torch.Tensor.size(c3))