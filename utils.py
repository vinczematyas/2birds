import torch
from torch.utils.data import Dataset
import numpy as np

class SignalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        return x, y

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'):
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

# define the loss function
def loss_function(recon_x, x, cond_data, mu, logvar, beta, wx, wy, fun_list):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    recon_loss_fn = torch.nn.L1Loss(reduction='mean')
    #recon_loss_fn = torch.nn.L1Loss(reduction='sum')
    #recon_loss_fn = torch.nn.MSELoss()
    x_loss  = recon_loss_fn(x, recon_x)

    # Calculate the next-wise-element functions in fun_list
    results_list = []
    x0 = recon_x[:,0,:,0].cpu().detach().numpy().flatten()
    x1 = recon_x[:,0,:,1].cpu().detach().numpy().flatten()

    for fun in fun_list:
        result = fun(x0, x1)
        results_list.append(result)

    Nw = recon_x.size(-2)
    recon_cond_data = np.vstack([results_list]).T.reshape(len(cond_data), Nw*len(fun_list))
    recon_cond_data = torch.Tensor(np.array(recon_cond_data)).type(torch.float)
    if torch.cuda.is_available():
        recon_cond_data = recon_cond_data.cuda()
    y_loss =  recon_loss_fn(cond_data, recon_cond_data.view(cond_data.shape))

    total_loss = (beta * KLD + wx * x_loss + wy * y_loss).mean()

    return total_loss, KLD, x_loss, y_loss