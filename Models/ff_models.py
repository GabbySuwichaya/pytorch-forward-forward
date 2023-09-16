import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import pdb

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_ 


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU() 

    def forward(self, x): 
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4) 
        output      = torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        output      = self.relu(output)  
        goodness    = output.pow(2).mean(1)
        return output, goodness
  

class Net(torch.nn.Module):

    def __init__(self, dims, num_class=10, lr=0.03, sub_epochs_per_layer=10):
        super().__init__() 

        self.layer_id = 0
        self.num_class = num_class

        self.threshold = 2.0
        self.sub_epochs_per_layer = sub_epochs_per_layer
        
        layers = []
        self.final_loss_per_layer = {}
        self.loss_per_layer = {}
        for d in range(len(dims) - 1):
            layers += [Layer(dims[d], dims[d + 1])]
            self.final_loss_per_layer["layer_%d" % d] = None
            self.loss_per_layer["layer_%d" % d]       = []

        self.layers = nn.ModuleList(layers).cuda()  
        self.opt = Adam(self.layers.parameters(), lr=lr)
    
    def generate_overlay_perlabel(self, x, y_index, x_max, num_class=10):  
        # Num label == Num batches == size x_max 
        x_    = x.clone()  
        x_[: , : num_class] *= 0.0   
        if x_.shape[0] > 1:
            x_[range(x_.shape[0]), y_index]      = x_max
        else: 
            x_[0, y_index]      = x_max
        return x_


    def preprocessing(self, x):
        x_max = x.max(dim=1).values 
        B, N  = x.shape # N = C x W X H  
        x_list = []
        y     = range(self.num_class)
        for y_index in y:
            x_temp = self.generate_overlay_perlabel(x, y_index, x_max, num_class=self.num_class)
            x_list.append(x_temp.view(B,1,-1))
        
        x_augment = torch.cat(x_list, dim = 1)
        return x_augment.view(B*self.num_class,-1)
    
    
    def preprocesing_training(self, x, y):  
        B, N   = x.shape # BxN = Bx[C x W X H]  
        x_max  = x.max(dim=1).values
        B      = y.shape
        
        permute_rand =  torch.randperm(x.size(0))
        y_neg        = y[permute_rand]
        # p       = torch.rand(B,1).to(y.device)  
        # y_rand_ = torch.randint(0, self.num_class, (B, 1)).to(y.device)    
        # y_neg   = y_rand_ + (p > 0.5)*(y_rand_ == y)*1.0 - (p <= 0.5)*(y_rand_ == y)*1.0    
        # y_neg   = (y_neg % self.num_class).int()
 
        # x_pos.shape = # BxN  
        x_pos = self.generate_overlay_perlabel(x, y,     x_max,  num_class=self.num_class)

        x_neg = self.generate_overlay_perlabel(x, y_neg, x_max,  num_class=self.num_class)
                                                
          
        return x_pos, x_neg
     
    def predict(self, x):
        goodness_per_label = []
        B, N  = x.shape # N = C x W X H  
        hidden = self.preprocessing(x) # h = [BxClass] x N  
        goodness_per_layer = []
        
        for layer in self.layers: 
            layer.eval()
            hidden, goodness = layer(hidden) # [BxClass] x N_out ==> [BxClass] x 1 
            goodness_per_layer.append(goodness) 
        
        goodness_BxC = goodness.view(B, self.num_class) # [BxClass] x 1 ===> [B]x[Class] 
        y_predict = goodness_BxC.argmax(1)   # [B]x[Class]   ===> [B]x1

        return y_predict # B x 1


    def train_per_layer(self, layer, x_pos, x_neg):  
        for i in range(self.sub_epochs_per_layer):
            y_pos , g_pos = layer(x_pos)
            y_neg , g_neg = layer(x_neg)
            
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            self.loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            self.loss.backward()
            self.opt.step()

            self.loss_per_layer["layer_%d" %  self.layer_id].append(self.loss.item())
        
        # Finish training per layer 
        out_pos, g_out_pos  = layer(x_pos)
        out_neg, g_out_neg  = layer(x_neg)
 
        loss_layer = torch.log(1 + torch.exp(torch.cat([
                -g_out_pos.detach()   + self.threshold,
                 g_out_neg.detach()   - self.threshold]))).mean() 
        self.final_loss_per_layer["layer_%d" %  self.layer_id] = loss_layer.item()
        
        # But you should check if [out_pos and neg_pos] == [g_pos, g_neg]
        return out_pos.detach(), out_neg.detach()  

    def train_process(self, x, y):
        x_pos, x_neg  = self.preprocesing_training(x, y)
        g_pos, g_neg  = x_pos, x_neg  
        for layer_id, layer in enumerate(self.layers):   
            self.layer_id = layer_id
            g_pos, g_neg  = self.train_per_layer(layer, g_pos, g_neg) 