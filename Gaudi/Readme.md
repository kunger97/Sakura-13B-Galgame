## Qwen2 OpenAI Compcompatible Server For Intel Gaudi

一个为Intel Gaudi硬件打造的以QWEN2为模型的OpenAI请求兼容服务器

经测试可以使用，但是不保证全部功能可用（推理参数）。您可以自行改造该程序。

本程序使用Huggingface的optimum库实现的大模型推理。具体加载模型代码参考utils.py，推理实现参考run_generation.py这两个代码来自optimum-habana库

```Text
https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation
```

具体参数请参考run_generation.py命令行中的帮助，并在server.py中的25行开始设定。

请您自行安装Gaudi硬件驱动和相关Python库，可以正常运行本程序的包列表如下：

```Text
Package                     Version
--------------------------- --------------------------
absl-py                     2.1.0
accelerate                  0.33.0
aiohappyeyeballs            2.3.6
aiohttp                     3.10.3
aiosignal                   1.3.1
async-timeout               4.0.3
attrs                       24.2.0
av                          9.2.0
blinker                     1.8.2
cachetools                  5.4.0
certifi                     2024.7.4
cffi                        1.15.1
cfgv                        3.4.0
charset-normalizer          3.3.2
click                       8.1.7
cmake                       3.30.2
coloredlogs                 15.0.1
datasets                    2.21.0
deepspeed                   0.14.0+hpu.synapse.v1.17.0
diffusers                   0.29.2
dill                        0.3.8
distlib                     0.3.8
exceptiongroup              1.2.2
expecttest                  0.2.1
filelock                    3.15.4
Flask                       3.0.3
Flask-Cors                  4.0.1
frozenlist                  1.4.1
fsspec                      2024.6.1
google-auth                 2.33.0
google-auth-oauthlib        0.4.6
grpcio                      1.65.5
habana_gpu_migration        1.15.0.479
habana-media-loader         1.15.0.479
habana-pyhlml               1.15.0.479
habana_quantization_toolkit 1.15.0.479
habana-torch-dataloader     1.15.0.479
habana-torch-plugin         1.15.0.479
hjson                       3.1.0
huggingface-hub             0.24.5
humanfriendly               10.0
identify                    2.6.0
idna                        3.7
importlib_metadata          8.2.0
iniconfig                   2.0.0
itsdangerous                2.2.0
Jinja2                      3.1.4
joblib                      1.4.2
lightning                   2.2.0.post0
lightning-habana            1.4.0
lightning-utilities         0.11.6
Markdown                    3.7
MarkupSafe                  2.1.5
mpi4py                      3.1.4
mpmath                      1.3.0
multidict                   6.0.5
multiprocess                0.70.16
networkx                    3.3
ninja                       1.11.1.1
nodeenv                     1.9.1
numpy                       1.23.5
oauthlib                    3.2.2
optimum                     1.21.4
optimum-habana              1.13.0
packaging                   24.1
pandas                      2.0.1
pathspec                    0.12.1
peft                        0.12.0
perfetto                    0.7.0
pillow                      10.4.0
Pillow-SIMD                 7.0.0.post3
pip                         22.0.2
platformdirs                4.2.2
pluggy                      1.5.0
pre-commit                  3.3.3
protobuf                    3.20.3
psutil                      6.0.0
py-cpuinfo                  9.0.0
pyarrow                     17.0.0
pyasn1                      0.6.0
pyasn1_modules              0.4.0
pybind11                    2.10.4
pycparser                   2.22
pydantic                    1.10.13
pynvml                      8.0.4
pytest                      8.3.2
python-dateutil             2.9.0.post0
pytorch-lightning           2.4.0
pytz                        2024.1
PyYAML                      6.0
regex                       2023.5.5
requests                    2.32.3
requests-oauthlib           2.0.0
rsa                         4.9
safetensors                 0.4.4
scikit-learn                1.5.1
scipy                       1.14.0
sentence-transformers       3.0.1
sentencepiece               0.2.0
setuptools                  72.2.0
six                         1.16.0
symengine                   0.11.0
sympy                       1.13.2
tdqm                        0.0.1
tensorboard                 2.11.2
tensorboard-data-server     0.6.1
tensorboard-plugin-wit      1.8.1
threadpoolctl               3.5.0
tokenizers                  0.19.1
tomli                       2.0.1
torch                       2.2.0a0+git8964477
torch_tb_profiler           0.4.0
torchaudio                  2.2.0+08901ad
torchdata                   0.7.1+5e6f7b7
torchmetrics                1.4.1
torchtext                   0.17.0+400da5c
torchvision                 0.17.0+b2383d4
tqdm                        4.66.5
transformers                4.43.4
typing_extensions           4.12.2
tzdata                      2024.1
urllib3                     1.26.19
virtualenv                  20.26.3
Werkzeug                    3.0.3
wheel                       0.44.0
xxhash                      3.4.1
yamllint                    1.35.1
yarl                        1.9.4
zipp                        3.20.0
```