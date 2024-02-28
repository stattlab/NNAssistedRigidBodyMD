import torch
import torch.nn as nn
import numpy as np
import sys,os
from torch.utils.data import DataLoader, Dataset,random_split
import time
import argparse
import re
import gc
# from Checker import CheckPoint
np.set_printoptions(precision=3, threshold=None, suppress=True)
torch.set_printoptions(precision=6)

"""
Jan 9

Experiment with batch size etc
"""


class MyDataset(Dataset):
    def __init__(self,filenames,target_column):
        self.x_data = torch.load(filenames[0])
        self.y_data = torch.load(filenames[1])
        shape = self.x_data.size()
        self.n_samples = shape[0]
        # self.y_data = self.y_data[:,target_column]
        self.y_data = torch.reshape(self.y_data,(self.n_samples,1))
        self.input_size = shape[1]
        self.normalize_input()


    def normalize_input(self):
        """
        Max-min normalization
        """
        # mins = [1.747,0.0,0.0,0.0,-3.141593,0.0]
        # maxs = [6.3,4.51,3.63,1.572, 3.141593,1.572]
        mins = [0.98,0.0,0.0,0.0,-3.142,0.0]
        maxs = [6.31,5.43,3.65,2.04, 3.142,1.572]
        N_features = self.x_data.size()[1]
        for i in range(N_features):
            data = torch.clone(self.x_data[:,i])
            # min = torch.min(data)
            # max = torch.max(data)
            # print(min,max)
            data_normal = (data - mins[i]) / (maxs[i] - mins[i])
            self.x_data[:,i] = data_normal
        # print("Fix this part next time ")
        # exit()


    def get_min_target(self):
        """
        to calculate relative error I must get rid of values zeros
        since they dominate the mean rel err so just when calculate
        rel err to print I add the minimum + 1.0 to all target and prediction
        """
        return torch.min(self.y_data).numpy()


    def load_to_gpu(self,device):
        self.x_data = self.x_data.to(device)
        self.y_data = self.y_data.to(device)

    def get_split_counts(self,fraction):
        N_test = self.n_samples*fraction
        N_test = int(N_test)
        N_train = self.n_samples - N_test
        return N_train,N_test

    def __getitem__(self,index):
        return self.x_data[index,:],self.y_data[index]

    def __len__(self):
        return self.n_samples

    def get_input_size(self):
        return self.input_size


class TheNet(nn.Module):
    """basic FF network for approximating functions"""
    def __init__(self, nn_width, num_hidden,input_size):
        super().__init__()

        self.layer_first = nn.Linear(input_size, nn_width)

        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)

        self.layer_last = nn.Linear(nn_width, 1)

    def forward(self, x):
        activation = nn.ReLU()
        u = activation(self.layer_first(x))
        for hidden in self.layer_hidden:
            u = activation(hidden(u))
        u = self.layer_last(u)
        return u

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_datas", nargs='+', required=True, help="-i x.pt y.pt" )
non_opt.add_argument('--lr', metavar="<float>", type=float, dest="lr", required=True, help="learning rate" )
non_opt.add_argument('--batch_size', metavar="<int>", type=int, dest="batch_size", required=True, help="run id" )
non_opt.add_argument('--width', metavar="<int>", type=int, dest="width", required=True, help="run id" )
non_opt.add_argument('--depth', metavar="<int>", type=int, dest="depth", required=True, help="run id" )
non_opt.add_argument('--gpu', metavar="<int>", type=int, dest="gpu", required=True, help="run id" )
non_opt.add_argument('--minutes', metavar="<int>", type=int, dest="minutes", required=False, help="run id", default=10000 )
non_opt.add_argument('--stop', metavar="<int>", type=int, dest="stopEpoch", required=True, help="run id" )
non_opt.add_argument('--col', metavar="<int>", type=int, dest="column", required=False, help="column to train 1 - 9 " )
non_opt.add_argument('--torfor', metavar="<int>", type=int, dest="torfor", required=True, help="0 for force, 1 for tork " )

args = parser.parse_args()
input_datas = args.input_datas
lr = args.lr
batch_size = args.batch_size
width = args.width
depth = args.depth
gpu = args.gpu
column = args.column
minutes = args.minutes
stopEpoch = args.stopEpoch
torfor = args.torfor
stop_seconds = minutes*60

if(torfor>0.5):
    torfor = 4

save_freq = 5

gpu_str = 'cuda:1'
if(gpu==0):
    gpu_str = 'cuda:0'

if not os.path.exists('./models/'):
    os.makedirs('./models/')

device = torch.device(gpu_str if torch.cuda.is_available() else 'cpu')

script_name = sys.argv[0]
script_name = script_name[:-3]

N_epochs = stopEpoch

train_dataset = MyDataset(input_datas,0)
y_min = train_dataset.get_min_target()
train_dataset.load_to_gpu(device)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
N_samples = train_dataset.__len__()
input_size = train_dataset.get_input_size()
model = TheNet(width,depth,input_size)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
epochs = []
print("Start Training ...")
t0 = time.time()
for i in range(N_epochs):
    model.train()
    train_loss = []
    for j,(X_batch,Y_batch) in enumerate(train_loader):
        y_hat = model.forward(X_batch)
        loss = criterion(y_hat,Y_batch)
        train_loss.append(loss.cpu().detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    t_loss = np.mean(np.array(train_loss))
    t1 = time.time()
    print("Epoch %d , error %.6f, seconds %.2f" %(i,t_loss,t1-t0))
    if i % save_freq == 0:
        # model_name = './models/model%d_NE%d.pth' %(column+torfor,i)
        # model_name = './models/model9_NE%d.pth' %i
        model_name = './models/model_bnkirk_NE%d_W%d_D%d.pth' %(i,width,depth)
        torch.save(model.state_dict(), model_name)
        if (t1-t0>stop_seconds):
            print("Training Completed.")
            exit()

















exit()
