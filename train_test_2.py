from Engine0 import get_mnist_data, MLP_X, Tensor, MLP_FLINEAR
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class Configs():
    batch_size: int
    learning_rate: float
    epochs: int
    layer_sizes: list


def smooth(values, window=20):
    '''Simple moving average, used only to make the loss curves readable.'''
    if len(values) < window:
        return values
    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]


def train_test_code(cfg: Configs, func_for_mnist_data):

    batch_size = cfg.batch_size
    train_data, test_data = func_for_mnist_data(batch_size)
    learning_rate = cfg.learning_rate
    epochs = cfg.epochs
    layer_sizes = cfg.layer_sizes

    # ---------------- Triton (basic, unfused) ----------------

    def train_triton_basic():
        model = MLP_X(layer_sizes)
        loss_history = []

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for epoch in range(epochs):
            for batch_idx, (x_batch, y_batch) in enumerate(train_data):
                y_one_hot = F.one_hot(y_batch, num_classes=10).float().to("cuda")
                x_tensor = Tensor(x_batch.to("cuda"))

                logits = model(x_tensor)
                loss = logits.softmax_entropy(y_one_hot)
                loss.backward()

                for p in model.parameters():
                    p.data -= learning_rate * p.grad
                    p.grad = torch.zeros_like(p.data)

                batch_loss = loss.data.mean().item()
                loss_history.append(batch_loss)

                if batch_idx % 100 == 0:
                    print(f"[Triton-Basic] Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss:.4f}")

        torch.cuda.synchronize()
        train_time = time.perf_counter() - t0

        return model, loss_history, train_time

    def test_triton_basic(model):
        count = 0
        total = 0
        for batch_idx, (x_batch, y_batch) in enumerate(test_data):
            x_tensor = Tensor(x_batch.to("cuda"))
            y_batch = y_batch.to("cuda")

            logits = model(x_tensor).data
            pred = torch.argmax(logits, dim=1)

            count += (pred == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = 100 * count / total
        print(f"[Triton-Basic] Inference -> correct = {count}, total = {total}, accuracy = {accuracy:.4f}%")
        return accuracy

    # ---------------- Triton (fused linear+relu) ----------------

    def train_triton_fused1():
        model = MLP_FLINEAR(layer_sizes)
        loss_history = []

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for epoch in range(epochs):
            for batch_idx, (x_batch, y_batch) in enumerate(train_data):
                y_one_hot = F.one_hot(y_batch, num_classes=10).float().to("cuda")
                x_tensor = Tensor(x_batch.to("cuda"))

                logits = model(x_tensor)
                loss = logits.softmax_entropy(y_one_hot)
                loss.backward()

                for p in model.parameters():
                    p.data -= learning_rate * p.grad
                    p.grad = torch.zeros_like(p.data)

                batch_loss = loss.data.mean().item()
                loss_history.append(batch_loss)

                if batch_idx % 100 == 0:
                    print(f"[Triton-Fused] Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss:.4f}")

        torch.cuda.synchronize()
        train_time = time.perf_counter() - t0

        return model, loss_history, train_time

    def test_triton_fused1(model):
        count = 0
        total = 0
        for batch_idx, (x_batch, y_batch) in enumerate(test_data):
            x_tensor = Tensor(x_batch.to("cuda"))
            y_batch = y_batch.to("cuda")

            logits = model(x_tensor).data
            pred = torch.argmax(logits, dim=1)

            count += (pred == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = 100 * count / total
        print(f"[Triton-Fused] Inference -> correct = {count}, total = {total}, accuracy = {accuracy:.4f}%")
        return accuracy

    # ---------------- PyTorch baseline ----------------

    def train_pytorch():

        class MLP_torch(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                for i in range(len(layer_sizes) - 1):
                    layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    if i < len(layer_sizes) - 2:
                        layers.append(nn.ReLU())
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        model = MLP_torch().cuda()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        loss_history = []

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for epoch in range(epochs):
            model.train()
            for batch_idx, (x_batch, y_batch) in enumerate(train_data):
                x_batch = x_batch.to("cuda")
                y_batch = y_batch.to("cuda")

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

                if batch_idx % 100 == 0:
                    print(f"[PyTorch] Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        torch.cuda.synchronize()
        train_time = time.perf_counter() - t0

        return model, loss_history, train_time

    def test_pytorch(model):
        count = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(test_data):
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)

                count += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = 100 * count / total
        print(f"[PyTorch] Inference -> correct = {count}, total = {total}, accuracy = {accuracy:.4f}%")
        return accuracy

    # ---------------- Run all three ----------------

    model_triton, loss_triton, time_triton = train_triton_basic()
    model_fused, loss_fused, time_fused = train_triton_fused1()
    model_torch, loss_torch, time_torch = train_pytorch()

    # ---------------- Inference timing (GPU-accurate, via CUDA events) ----------------

    def time_inference(test_fn, model):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        acc = test_fn(model)
        end_evt.record()
        torch.cuda.synchronize()

        return acc, start_evt.elapsed_time(end_evt)  # milliseconds

    acc_triton, inf_time_triton = time_inference(test_triton_basic, model_triton)
    acc_fused, inf_time_fused = time_inference(test_triton_fused1, model_fused)
    acc_torch, inf_time_torch = time_inference(test_pytorch, model_torch)

    # ---------------- Summary table ----------------

    print("\n" + "=" * 62)
    print(f"{'Model':<15}{'Train Time (s)':<18}{'Inference (ms)':<18}{'Accuracy (%)':<12}")
    print("=" * 62)
    print(f"{'Triton-Basic':<15}{time_triton:<18.3f}{inf_time_triton:<18.3f}{acc_triton:<12.2f}")
    print(f"{'Triton-Fused':<15}{time_fused:<18.3f}{inf_time_fused:<18.3f}{acc_fused:<12.2f}")
    print(f"{'PyTorch':<15}{time_torch:<18.3f}{inf_time_torch:<18.3f}{acc_torch:<12.2f}")
    print("=" * 62 + "\n")

    # ---------------- Plots ----------------

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = ["Triton-Basic", "Triton-Fused", "PyTorch"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # 1) Training loss curves
    axes[0, 0].plot(smooth(loss_triton), label="Triton-Basic")
    axes[0, 0].plot(smooth(loss_fused), label="Triton-Fused")
    axes[0, 0].plot(smooth(loss_torch), label="PyTorch")
    axes[0, 0].set_title("Training Loss (moving avg)")
    axes[0, 0].set_xlabel("Batch iteration")
    axes[0, 0].set_ylabel("Cross-entropy loss")
    axes[0, 0].legend()

    # 2) Training time only
    train_times = [time_triton, time_fused, time_torch]
    axes[0, 1].bar(labels, train_times, color=colors)
    axes[0, 1].set_title("Training Time")
    axes[0, 1].set_ylabel("Seconds")

    # 3) Inference time only (kept in ms, since it's much smaller than train time)
    inf_times_ms = [inf_time_triton, inf_time_fused, inf_time_torch]
    axes[1, 0].bar(labels, inf_times_ms, color=colors)
    axes[1, 0].set_title("Inference Time (full test set)")
    axes[1, 0].set_ylabel("Milliseconds")

    # 4) Test accuracy
    accs = [acc_triton, acc_fused, acc_torch]
    axes[1, 1].bar(labels, accs, color=colors)
    axes[1, 1].set_title("Test Accuracy")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylim(0, 100)

    plt.tight_layout()
    out_path = "comparison_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    cfg = Configs(batch_size=32, learning_rate=0.01, epochs=3, layer_sizes=[784, 128, 64, 10])
    train_test_code(cfg, get_mnist_data)