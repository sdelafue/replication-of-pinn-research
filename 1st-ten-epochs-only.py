from model1_multi import define_net, my_dataset
import torch
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
from torchsummary import summary


import warnings
warnings.filterwarnings("ignore")

def read_data(var, num_begin, num_end):
    """
    This function reads (loads) the npy file which in this case are vx, vy, and density files.
    
    
    Parameters
    ----------
    var : string
        It represents the identifier in the file name: for example "den" for the density.
    num_begin : int
        Minimum Image ID 
    num_end : int
        Maximum Image ID 
        
        
    Returns
    -------
    Returns a tensor with the size (No. of images, horizontal image size, vertical image size, No. of channels 
                                    for each image i.e. 1 for black and white)
    as well as the minumum and maximum value of the tensor.

    """
    Min=[]
    Max=[]

    den1 = np.load('datasets/'+var+'.npy')
    Min.append(den1.min())
    Max.append(den1.max())
    den1 = den1[num_begin:num_end]

    Den = den1
    return Den, max(Max), min(Min);
    
#%%
n = 128
den = np.load('datasets/'+'den'+'.npy')


vx = np.load('datasets/'+'vx'+'.npy')
vx_min = vx.min()
vx_max = vx.max()

vy = np.load('datasets/'+'vy'+'.npy')
vy_min = vy.min()
vy_max = vy.max()
den = np.around(den)
vy    = (vy - vy_min)/(vy_max - vy_min)
vx     = (vx - vx_min)/(vx_max - vx_min)

#%%
plt.imshow(den[0,...,0])
plt.imshow(vx[0,...,0])
ratio = 0.00056174e-2
ratio2 = 5e-8
ratioy = 0.00031435e-2
ratio2y = 5e-8


ratio_g = 0

#%%

isTrain = True

model_name = 'URESNET'  # RESNET, UNET, URESNET
input_nc   = 1
output_nc  = 1
gpu_ids    = []  # Run on CPU
lr         = 0.0002
tot_var_weight = 0.0
batch_size = 16
Gradientxyx = 0
Gradientxy = 0

ngf        = 8
torch.autograd.set_detect_anomaly(True)
path = './weights_new/'  # Use a local path for weights
ep_num = 150

train_sample_num = 1200
val_sample_num   = 600

continue_train = False
continue_epoch = 50

netx = define_net(input_nc, output_nc, ngf=ngf, gpu_ids=gpu_ids, model_name=model_name)
nety = define_net(input_nc, output_nc, ngf=ngf, gpu_ids=gpu_ids, model_name=model_name)

Size = 128

import os
if not os.path.exists(path+'weights'):
    os.makedirs(path+'weights', exist_ok=True)
weights_filename = path+'weights/weights'

if continue_train:
    netx.load_state_dict(torch.load(weights_filename + 'xg_{}.pt'.format(continue_epoch), map_location=torch.device('cpu')))
    nety.load_state_dict(torch.load(weights_filename + 'yg_{}.pt'.format(continue_epoch), map_location=torch.device('cpu')))

if isTrain:
    old_lr = lr

    # define loss functions

    criterion = torch.nn.L1Loss()
    # initialize optimizers
    optimizery = torch.optim.Adam(list(nety.parameters()) + list(netx.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.00150)

##%% 
dset         = my_dataset(path, train_sample_num, val_sample_num, input_file=den, output1_file=vx, output2_file=vy, data_mode = 'both', train_val_test=0 )


train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)

min_lossx = 1e5
min_lossy = 1e5


train_lossesx = []
train_lossesy = []
val_lossesx = []
val_lossesy = []
tot_varx_2_total = 0

#%% Training the model
tot_vary_2_total = 0

# Lists to collect predictions

# Only collect predictions for the first 10 epochs
all_predicted_x = []
all_predicted_y = []

for ep in range(ep_num):
    print('ep #{}'.format(ep))
    train_lossx_total = 0
    train_lossy_total = 0
    netx.train()
    nety.train()
    for img_in, img_out_x, img_out_y in train_loader:
        img_in, img_out_y, img_out_x = Variable(img_in), Variable(img_out_y), Variable(img_out_x)
        optimizery.zero_grad()
        # ...existing code for forward, loss, backward, optimizer step...
        recx = netx.forward(img_in)
        recy = nety.forward(img_in)
        # ...existing code for loss calculation...
        # ...existing code for backward and optimizer step...
        # ...existing code for loss accumulation...
        # ...existing code for tot_varx_2_total, tot_vary_2_total...
        # ...existing code for loss averaging...
        # ...existing code for validation...
        # ...existing code for val_lossx_total, val_lossy_total...
        # ...existing code for loss calculation in validation...
    if ep < 10:
        netx.eval()
        nety.eval()
        for img_in, img_out_x, img_out_y in train_loader:
            img_in, img_out_y, img_out_x = Variable(img_in), Variable(img_out_y), Variable(img_out_x)
            recx = netx.forward(img_in)
            recy = nety.forward(img_in)
            all_predicted_x.append(recx.detach().cpu().numpy())
            all_predicted_y.append(recy.detach().cpu().numpy())
    if ep == 9:
        # Save predictions for the first 10 epochs only
        if all_predicted_x:
            all_predicted_x_np = np.concatenate(all_predicted_x, axis=0)
            np.save('all_predicted_x.npy', all_predicted_x_np)
        if all_predicted_y:
            all_predicted_y_np = np.concatenate(all_predicted_y, axis=0)
            np.save('all_predicted_y.npy', all_predicted_y_np)
        import sys
        sys.exit(0)


np.savez('div2valtrain.npz', np.array(val_lossesx), np.array(train_lossesx))
np.savez('divtrainlearning_ratemodel1multi_{}.npz'.format(str(ratio)), np.array(train_lossesx), np.array(train_lossesx))


#%%
