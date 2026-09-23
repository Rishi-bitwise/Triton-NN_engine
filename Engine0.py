from Tkernels import TMUL, TADD, TSFX, THAD, TRELU, TMATXADD
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import math

#TODO REPLACE ALL += OPS FROM PYTORCH TO TRITON FUSED ??

DEVICE = torch.device("cuda")

class Tensor():
    def __init__(self, data: torch.Tensor, children=None, op=""):
        
        self.rows = data.shape[0]
        self.cols = 1
        if len(data.shape) > 1:
            self.cols = data.shape[1]

        else:
            data = data.reshape([-1, 1])
            self.cols = 1


        data = data.to(device=DEVICE)    #this is torch.Tensor type
        self.data = data
        self.shape = data.shape
        
        self.children = children if children is not None else ()
        self.op = op
        self.grad = torch.zeros_like(data)
        self.back = lambda : None

    def numel(self):
        return self.rows*self.cols


    def matmul(self, other):
        out_data = TMUL(self.data, other.data)

        out = Tensor(out_data, children=(self, other), op="@")

        def back():
            self.grad += TMUL(out.grad, other.data, trans_b=True)
            other.grad += TMUL(self.data, out.grad, trans_a=True)

        out.back = back

        return out

    def add(self, other):
        out_data = TADD(self.data, other.data)    #TADD handles broadcasting
        out = Tensor(out_data, children=(self, other), op="+")

        def back():
            if self.grad.shape[0] == 1 and out.grad.shape[0] > 1:
                self.grad += out.grad.sum(dim=0, keepdim=True)              #TODO pytorch used here

            else:
                self.grad += out.grad

            if other.grad.shape[0] == 1 and out.grad.shape[0] > 1:          #TODO pytorch is used here. 
                other.grad += out.grad.sum(dim=0, keepdim=True)

            else:
                other.grad += out.grad

        out.back = back 
        return out

    def matxadd(self, other1, other2):
        '''self@other1 + other2'''
        out_data, relu_derivative = TMATXADD(self.data, other1.data, other2.data)

        out = Tensor(out_data, children = (self, other1, other2), op="@+")

        def back():
            out_grad_with_relu = THAD(out.grad, relu_derivative)
            self.grad += TMUL(out_grad_with_relu, other1.data, trans_b=True)
            other1.grad += TMUL(self.data, out_grad_with_relu, trans_a=True)

            if other2.grad.shape[0] == 1 and out.grad.shape[0] > 1:          #TODO pytorch is used here. 
                other2.grad += out_grad_with_relu.sum(dim=0, keepdim=True)

            else:
                other2.grad += out_grad_with_relu


        out.back = back

        return out
    # def sub(self, other):
    #     out_data = TSUB(self.data, other.data)
    #     out = Tensor(out_data, children=(self,other), op="-")

    #     def back():
    #         self.grad += out.grad
    #         other.grad -= out.grad

    #     out.back = back
    #     return out

    def softmax_entropy(self, Y:torch.Tensor):
        '''Note it is assumed that self here is the output_layer. This func gives the final loss and has nothing else after it. 
        Returns loss as 1D and P as 2D. Y here is one hot encoded, so its 2D, likely (32x10)'''

        out_data, P = TSFX(self.data, Y)   #out_data is Bx1 (i think thats what its supposed to be)

        P = Tensor(P, children=(self,), op="SOFT")

        out = Tensor(out_data, children=(self,), op="X")

        def back():
            batchsize = self.grad.shape[0]
            grad_matrix = (P.data-Y)/batchsize              #TODO pytorch is being used here, maybe change it to fused triton only...
            self.grad +=  grad_matrix                       # Here loss.backward sets the out_layer grads. then since out_layer(self) is in children we backprop again
        
        out.grad = torch.ones_like(out.data)                #TODO This is unnecessary, but i dont wanna break anything
        out.back = back
        
        return out


    def relu(self):
        out_data, grad = TRELU(self.data)
        out = Tensor(out_data, children=(self,), op="relu")

        def back():
            self.grad += THAD(out.grad, grad)           #for hadamard product with the gradient, which is 0 for negs, and 1 for pos vals

        out.back = back
        return out


    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v in visited:
                return

            else:
                visited.add(v)
                for p in v.children:
                    build(p)

                topo.append(v)

        build(self)
        self.grad = torch.ones_like(self.data)

        for v in reversed(topo):
            v.back()


class Linear():
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        bound = 1.0 / math.sqrt(in_features)
        self.W = Tensor(torch.empty(in_features, out_features).uniform_(-bound, bound), op="W_init")
        # B: (1, out_features) - this broadcasts across the batch during addition
        self.B = Tensor(torch.empty(1, out_features).uniform_(-bound, bound), op="B_init")

    def __call__(self, x: Tensor):
        return x.matmul(self.W).add(self.B)

    def parameters(self):
        return [self.W, self.B]

class FLinear():
    '''Same as linear, just has the fused kernel'''
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        bound = 1.0/math.sqrt(in_features)
        self.W = Tensor(torch.empty(in_features, out_features).uniform_(-bound, bound), op="W_init")
        self.B = Tensor(torch.empty(1, out_features).uniform_(-bound, bound), op="B_init")

    def __call__(self, x:Tensor):
        return x.matxadd(self.W, self.B)

    def parameters(self):
        return [self.W, self.B]



class MLP_X():
    '''X is for softmax + xentropy'''
    def __init__(self, layer_sizes):
        # layer_sizes is a list, e.g., [784, 128, 10] for MNIST
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))

    def __call__(self, x: Tensor):
        # Forward pass
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out) # Applies X @ W + B
            
            # Apply ReLU to every layer except the last one
            if i < len(self.layers) - 1:
                out = out.relu()
                
        return out

    def parameters(self):
        params = []                 #Contains list of weights and bias Tensors. 
        for layer in self.layers:
            params.extend(layer.parameters())
        return params



class MLP_FLINEAR():
    '''FLINEAR meaning it uses the linear class that has fused kernels, except last layer which is matmul+add'''
    def __init__(self, layer_sizes):
        # layer_sizes is a list, e.g., [784, 128, 10] for MNIST
        self.layers = []
        for i in range(len(layer_sizes) - 2):           #We skip the fused nonsense for the last layer, since this is for MNIST, no activation for last layer
            self.layers.append(FLinear(layer_sizes[i], layer_sizes[i+1]))

        self.layers.append(Linear(layer_sizes[-2], layer_sizes[-1]))

        
    def __call__(self, x: Tensor):
        # Forward pass
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out) # Applies RELU(X @ W + B) (fused), except for the last layer which is a simple non fused matmul_add layer
        
        return out

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params



def get_mnist_data(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

    return train_loader, test_loader





