![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)

这个仓库涵盖了论文的主要代码。包括如下：

本论文提出的模型**ODEformer**

**实验一：基线比较实验**
Autoformer:

Informer:

Pyraformer:

Reformer:

此外特别感谢对本文代码实现的启发。

**实验结果：**

<img width="418" alt="1680970472475" src="https://user-images.githubusercontent.com/91870223/230731713-5685e857-fdd7-4d74-a742-c4c64d93a602.png">

**实验二：组件消融实验**（Trans_embed、Trans_spilt）

**NODE-embed 组件：**

![image](https://user-images.githubusercontent.com/91870223/230731143-36b64d1a-4d25-4448-9d0f-0774ffd60232.png)


**NODE-spilt组件：**

![image](https://user-images.githubusercontent.com/91870223/230731115-3bb4370b-baeb-4163-804e-81805cf1f5c0.png)

**实验结果：**

<img width="446" alt="1680970020228" src="https://user-images.githubusercontent.com/91870223/230731374-6ce14aa1-620b-4066-9a19-8ac989a5ad74.png">


**实验三：NODE验证实验**(try_withoutCNN、try_withoutTCN)

<img width="438" alt="1680970558549" src="https://user-images.githubusercontent.com/91870223/230731779-faf5a087-42d0-4b99-b8ae-72d0dc6b01d8.png">

