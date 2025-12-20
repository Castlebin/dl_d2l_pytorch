# dl_d2l

This repository contains PyTorch implementations of the Deep Learning book "Dive into Deep Learning" (D2L).   
It provides code examples, tutorials, and exercises to help you learn deep learning concepts using PyTorch.

## Installation

```
pip install -U dl_d2l
```

## Code Examples

You can find the code examples in the `dl_d2l` package. Here is a simple example of how to use it:

```python
from dl_d2l import d2l_torch as d2l


```



## Build & upload to pypi (For Developers)

prerequirement: twine is installed. If not, run the following command to install it:

```bash
pip install -U twine
```

build and upload:

```bash
## package
python setup.py sdist

## upload
twine upload dist/*
```

## D2L
For more information about the "Dive into Deep Learning" book, visit the [official website](https://d2l.ai/).


