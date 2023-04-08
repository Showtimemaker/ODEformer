![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

# 基于神经常微分方程的时间序列多尺度特征提取、融合与预测

这个仓库涵盖了论文的主要代码。包括论文提出的模型**ODEformer**以及相关实验。

## **实验一：基线比较实验**

  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py)
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py)
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py)
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py)
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)


此外特别感谢[[Time-Series-Library]](https://github.com/thuml/Time-Series-Library)对本文代码实现的启发。

**实验结果：**

<img width="418" alt="1680970472475" src="https://user-images.githubusercontent.com/91870223/230731713-5685e857-fdd7-4d74-a742-c4c64d93a602.png">
<img width="420" alt="1680970656134" src="https://user-images.githubusercontent.com/91870223/230731852-0280b625-8b1e-437f-b189-0f7406d24e19.png">
<img width="418" alt="1680970683705" src="https://user-images.githubusercontent.com/91870223/230731870-dedc9833-1efa-46c4-80ae-6c60b8b5f1a3.png">


## **实验二：组件消融实验**（Trans_embed、Trans_spilt）

**NODE-embed 组件：**

![image](https://user-images.githubusercontent.com/91870223/230731143-36b64d1a-4d25-4448-9d0f-0774ffd60232.png)


**NODE-spilt组件：**

![image](https://user-images.githubusercontent.com/91870223/230731115-3bb4370b-baeb-4163-804e-81805cf1f5c0.png)

**实验结果：**

<img width="446" alt="1680970020228" src="https://user-images.githubusercontent.com/91870223/230731374-6ce14aa1-620b-4066-9a19-8ac989a5ad74.png">


## **实验三：NODE验证实验**(try_withoutCNN、try_withoutTCN)

<img width="438" alt="1680970558549" src="https://user-images.githubusercontent.com/91870223/230731779-faf5a087-42d0-4b99-b8ae-72d0dc6b01d8.png">

