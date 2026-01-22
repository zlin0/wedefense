# Installation

This guide will help you install **wedefense** and its dependencies.

## Install via pip

*Coming soon.*
(Currently, pip installation is not supported or is under development. Please use the local installation method below.)

## Install locally

We assume you have installed Python and [conda](https://docs.conda.io/).
You can set up the environment for wedefense as follows:
```shell
git clone https://github.com/zlin0/wedefense.git
cd wedefense
bash ./install_env.sh
```

After installation, activate the environment:
```shell
conda activate wedefense
```

## Test your GPU installation

To verify your installation and check if your GPU is available, run the following in a Python shell:
```python
import torch
print(torch.cuda.is_available())
```
If it prints `True`, your GPU setup is working.


- For more help, see the [FAQ](faq.md) or open an issue on [GitHub](https://github.com/zlin0/wedefense/issues).
