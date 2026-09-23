from Engine0 import get_mnist_data, MLP_X, Tensor, MLP_FLINEAR
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from dataclasses import dataclass

DEVICE = torch.device("cuda")

@dataclass
class HyperParams():
    batch_size : int
    learning_rate : float
    epochs : int
    layer_sizes : list


def train_test_code(cfg:HyperParams, func_for_mnist_data):

    batch_size = cfg.batch_size
    train_data, test_data = func_for_mnist_data(batch_size)
    learning_rate = cfg.learning_rate
    epochs = cfg.epochs
    layer_sizes = cfg.layer_sizes


    def train_triton_basic()->MLP_X:
        
        model = MLP_X(layer_sizes)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_idx, (x_batch, y_batch) in enumerate(train_data):
                # x_batch shape: (32, 784), y_batch shape: (32,)
                
                # Convert integer labels to one-hot encoded vectors. Shape becomes (32, 10)
                y_one_hot = F.one_hot(y_batch, num_classes=10).float().to(DEVICE)
                
                # Wrap the input data in your custom Tensor class
                x_tensor = Tensor(x_batch.to(DEVICE))
                
                # --- FORWARD PASS ---
                #  MLP_X.__call__
                logits = model(x_tensor)
                
                # Connects to Tensor.softmax_entropy
                # Note: your softmax_entropy expects Y as a torch.Tensor, not your custom Tensor
                loss = logits.softmax_entropy(y_one_hot)
                
                loss.backward()
                
                # --- OPTIMIZER (Stochastic Gradient Descent) ---
                # Connects to MLP_X.parameters() -> Linear.parameters()
                for p in model.parameters():
                    # Update the raw torch.Tensor data using the computed gradients
                    p.data -= learning_rate * p.grad
                    
                    # Zero the gradients for the next iteration
                    p.grad = torch.zeros_like(p.data)
                    
                total_loss += loss.data.mean().item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.data.mean().item():.4f}")

        return model

    def test_triton_basic(model):
        
        count = 0
        total = 0
        for batch_idx , (x_batch, y_batch) in enumerate(test_data):
            x_tensor = Tensor(x_batch.to(DEVICE))
            y_one_hot = F.one_hot(y_batch, num_classes=10).float().to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(x_tensor)

            logits = logits.data

            pred = torch.argmax(logits, dim=1)
            count += (pred == y_batch).sum().item()

            total += y_batch.size(0)
        accuracy = 100 * count/total
        print(f"Inference for TritonBasic ->\n correct predictions = {count}, total = {total}, percentage = {accuracy :.4f}")

    def train_triton_fused1()->MLP_FLINEAR:

        model = MLP_FLINEAR(layer_sizes)

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (x_batch, y_batch) in enumerate(train_data):
                y_one_hot = F.one_hot(y_batch, num_classes=10).float().to(DEVICE)
                x_tensor = Tensor(x_batch.to(DEVICE))

                logits = model(x_tensor)

                loss = logits.softmax_entropy(y_one_hot)

                loss.backward()

                for p in model.parameters():
                    p.data -= learning_rate*p.grad

                    p.grad = torch.zeros_like(p.data)

                total_loss += loss.data.mean().item()

                if batch_idx % 100 ==0:
                    print(f"TFused Epoch {epoch}, Batch {batch_idx}, Loss: {loss.data.mean().item():.4f}")

        return model

    def test_triton_fused1(model):
        count = 0
        total = 0

        for batch_idx, (x_batch, y_batch) in enumerate(test_data):
            x_tensor = Tensor(x_batch.to(DEVICE))
            y_one_hot = F.one_hot(y_batch, num_classes=10).float().to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(x_tensor)
            logits = logits.data

            pred = torch.argmax(logits, dim=1)
            count += (pred==y_batch).sum().item()

            total += y_batch.size(0)

        accuracy = 100*count/total
        print(f"Inference for TritonFused ->\n correct predictions = {count}, total = {total}, percentage = {accuracy :.4f}")


    def train_pytorch():

        class MLP_torch(nn.Module):

            def __init__(self):
                super().__init__()

                self.network = nn.Sequential(
                    nn.Linear(784, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10)
                )


            def forward(self, x):
                return self.network(x)


        
        def train_model_pytorch():
            model = MLP_torch().to(device=DEVICE)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):

                model.train()
                total_loss = 0.0

                for batch_idx, (x_batch, y_batch) in enumerate(train_data):
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    total_loss += loss.item()

                    if batch_idx % 100 == 0:
                        print(f"Epoch {epoch}, Batch {batch_idx}, "f"Loss: {loss.item():.4f}")
            return model

        model = train_model_pytorch()
        return model


    def test_pytorch(model):

        count = 0
        total = 0

        with torch.no_grad():
            for batch_idx , (x_batch, y_batch) in enumerate(test_data):
                x_batch, y_batch = x_batch.to(device=DEVICE), y_batch.to(device=DEVICE)
                logits = model(x_batch)

                preds = torch.argmax(logits, dim=1)

                count += (preds == y_batch).sum().item()
                total += y_batch.size(0)


        accuracy = 100 * count/total
        print(f"Inference for Pytorch ->\n correct predictions = {count}, total = {total}, percentage = {accuracy :.4f}")


    model_triton = train_triton_basic()
    model_triton_fused1 = train_triton_fused1()
    model_pytorch = train_pytorch()

    triton_basic_start = torch.cuda.Event(enable_timing=True)
    triton_basic_end = torch.cuda.Event(enable_timing=True)

    triton_f1_start = torch.cuda.Event(enable_timing=True)
    triton_f1_end = torch.cuda.Event(enable_timing=True)

    pytorch_start = torch.cuda.Event(enable_timing=True)
    pytorch_end = torch.cuda.Event(enable_timing=True)

    triton_basic_start.record()
    test_triton_basic(model_triton)
    triton_basic_end.record()

    torch.cuda.synchronize()

    triton_basic_inf_time = triton_basic_start.elapsed_time(triton_basic_end)

    triton_f1_start.record()
    test_triton_fused1(model_triton_fused1)
    triton_f1_end.record()

    triton_f1_inf_time = triton_f1_start.elapsed_time(triton_f1_end)

    pytorch_start.record()
    test_pytorch(model_pytorch)
    pytorch_end.record()

    torch.cuda.synchronize()

    pytorch_inf_time = pytorch_start.elapsed_time(pytorch_end)


    print(f"The total time for inference by \ntriton_basic = {triton_basic_inf_time}\n")
    print(f"The total time for inference by \ntriton_fused1 = {triton_f1_inf_time}\n")
    print(f"The total time for inference by \nPytorch = {pytorch_inf_time}\n")



if __name__ == "__main__":
    cfg = HyperParams(batch_size=32, learning_rate=0.01, epochs=5, layer_sizes=[784, 128, 64, 10])
    train_test_code(cfg, get_mnist_data)