# Needle — Necessary Elements of Deep Learning

**Needle** is a minimalist, educational deep learning framework built from scratch with a focus on clarity, modularity, and flexibility. It offers the **necessary elements** of deep learning—no more, no less.

Whether you're a student learning how autograd works, a developer exploring backprop under the hood, or a researcher prototyping quickly, Needle gives you the tools to build and train neural networks with full transparency.

---

## ✨ What is Needle?

Needle stands for:

> **Ne**cessary **E**lements of **D**eep **Le**arning

You can also imagine it as a **sewing needle** that threads through fabric to form (neural) net patterns—creating clean traces for automatic differentiation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61396368/197850311-75937074-873e-4e3f-a552-b67837d5dce7.png" width="500">
</p>

---

## 🚀 Features

- ✅ **Tensors** with broadcasting, operations, and NumPy-style syntax
- 🔁 **Autograd** engine supporting backward mode differentiation
- 🧠 **Neural Network modules** (`nn.Linear`, `nn.ReLU`, etc.)
- 📦 **Optimizers** like SGD
- 🔍 Minimal dependencies, readable source code
- 📚 Perfect for learning or extending deep learning internals

---

## 📦 Installation

```bash
git clone https://github.com/mohamed-seyam/needle.git
cd needle
pip install -e .
```

---

## 🛠️ Quickstart

Here are a few examples to help you get started with Needle.

### 1. Basic Tensor Operations

```python
import needle as ndl

a = ndl.Tensor([[1.0, 2.0], [3.0, 4.0]])
b = ndl.Tensor([[5.0, 6.0], [7.0, 8.0]])

print("Addition:\n", (a + b).numpy())
print("Matrix Multiplication:\n", (a @ b).numpy())

x = ndl.Tensor([3.0, 2.0], requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print("Gradients:\n", x.grad)
```

---

### 2. Building a Simple Neural Network

```python
import needle as ndl
import needle.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(nn.relu(self.fc1(x)))

model = SimpleNN(2, 4, 1)
input_data = ndl.Tensor([[0.5, -1.5], [2.0, 3.0]])
output = model(input_data)
print("Output:\n", output.numpy())
```

---

### 3. Using Autograd

```python
import needle as ndl

x = ndl.Tensor([3.0], requires_grad=True)
f = x**3 + 2 * x**2 + x
f.backward()
print("f(x):", f.numpy())
print("df/dx:", x.grad)
```

---

## 🧠 Codebase Overview

```
needle/
├── autograd.py      # Core autograd engine
├── tensor.py        # Tensor implementation
├── ops.py           # Basic operations with gradients
├── nn/              # Neural network modules
│   └── modules.py
├── optim.py         # Optimizers like SGD
├── data/            # (Optional) Dataset utils
└── examples/        # Tutorial scripts
```

---

## 🧪 Try It Out

We recommend starting with the examples in the [`examples/`](./examples) folder or checking out the [walkthrough](#) (coming soon!).

---

## 📚 Learn More

- [Deep Learning Framework from Scratch – A Walkthrough](https://github.com/mohamed-seyam/needle)
- Educational materials and future blog posts (coming soon)
- Stay tuned for support for convolution, batch norm, and more!

---

## 🤝 Contributing

Contributions, issues, and ideas are welcome! To contribute:

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a pull request 🚀

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## ⭐️ Show Your Support

If you like this project, consider giving it a star ⭐ on GitHub to help others discover it!
