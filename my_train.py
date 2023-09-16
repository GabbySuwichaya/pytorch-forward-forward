import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam  
from torch.utils.data import DataLoader

from Models.ff_models import Net
from Data.customized_MNIST import CustomMNISTDataset , MNIST_loaders
    
import pdb
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__": 

    # My adjustment: 
    # sampling_neg == 'rand' is the new option that I create to randomly generate the GT negative sample. 
    # This is because the original "perm" option forces us to use the high batch size for high variety when sampling. 
    # 
    # It turns out that this process reduce the error at the lower batch size (from 0.8 to about 0.08)
    #
    # More layers ==> Higher error 
    # Lower batch size ==> Higher error
    # lower sub_epoch  ==> Higher error 
    # Higher learning rate ==> Higher error
    
    batch_size = 1024
    sub_epoch  = 1000
    lr         = 0.01
    num_layers = 3

    sampling_neg = "rand" # | "perm"

    if num_layers == 2:
        net_list = [784, 500, 500]
    
    elif num_layers == 3:
        net_list = [784, 500, 500, 500]
    
    elif num_layers == 4:
        net_list = [784, 500, 500, 500, 500]


    # ===========================================================================================================================

    if lr == 0.03:
        checkpoint_path = "./weights/FF-MNIST-%s-%dlayers-%d-%d.pth" % (sampling_neg, num_layers, batch_size, sub_epoch, lr)
    else:
        checkpoint_path = "./weights/FF-MNIST-%s-%dlayers-%d-%d-%f.pth" % (sampling_neg, num_layers,  batch_size, sub_epoch, lr)

 
    net = Net(net_list, num_class=10, lr=lr, sub_epochs_per_layer=sub_epoch, sampling_neg=sampling_neg) 

    training = True

    if training: 
        net.train()

        #train_loader, test_loader = MNIST_loaders()
        MNIST_ = CustomMNISTDataset(train=True)
        train_loader = DataLoader(MNIST_, batch_size=batch_size, shuffle=True) 
        
        pbar = tqdm(train_loader)
        
        loss_epoch = []
        for epoch, data in enumerate(pbar): 
            x, y = data
            x, y = x.cuda(), y.cuda()
            
            x_pos, x_neg   = net.preprocesing_training(x, y)
            g_pos, g_neg   = x_pos, x_neg  

            loss_layer_dict = {}
    

            for layer_id, layer in enumerate(net.layers):
                
                net.layer_id = layer_id 
    
                g_pos, g_neg  = net.train_per_layer(layer, g_pos, g_neg)  
    
                loss_layer = net.final_loss_per_layer["layer_%d" % layer_id]

                loss_layer_dict["layer_%d" % layer_id] = loss_layer

            
            loss_epoch.append(loss_layer_dict)
            pbar.set_postfix(loss_layer_dict)

            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': net.opt.state_dict(),
                    'loss': net.loss
                }
        
        torch.save(checkpoint, checkpoint_path)

  

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()
 

    y_predict_list = []
    y_gt_list = []
    y_hit = []

    MNIST_test = CustomMNISTDataset(train=False)
    test_loader = DataLoader(MNIST_test, batch_size=1, shuffle=True) 
    
        
    for i, data in enumerate(test_loader):
        x_test, y_test = data
        x_test, y_test = x_test.cuda(), y_test.cuda()
        with torch.no_grad():
            y_predict      = net.predict(x_test)    

        
        y_predict_list.append(y_predict)
        y_gt_list.append(y_test)

        y_hit.append( 1*(y_test == y_predict) )
 
    print("Test error: %f" % (1 - sum(y_hit)/len(y_hit))) 