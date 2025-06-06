# HPC_in_EdgeDevice
This Repo records my personal learning in AI_deployed_on_EdgeDevice.
## 一、Ampere（A100） 架构学习（Orin 用的GPU同为Ampere架构）
每个A100 有108个SM（流式多处理器），每个SM有64个Cuda核心（int32，fp32），一个block最多对应对应1024个线程（这个是硬件当中预先定义好的，无法改变，所以cuda编程时，每个block的线程设置不能超过1024个），多个block对应我们A100架构的一个SM处理单元，block被分到某个SM上，则会保存到该SM上直到执行解说，同一时间段一个SM可以同时容纳多个block，每个SM中有1024个FMA独立计算单元，对应2048个独立的浮点运算，等效为2048个线程（这里不是SM的cuda core总数，而是最大活跃线程，即一个时钟周期可以执行2048个线程，block内线程的个数设置成1024，即最大活跃线程的一半），至于为什么是2048，因为一个SM有4个warp scheduler，最多能同时管理 64 个 warps（64*32=2048）A100总共108个SM，所以A100总共存在108*2048=221184个并发线程（最大活跃线程）。
所以cuda编程当中，一个block只能写1024个线程即：

dim3 blockdim(1024,1,1)或者
dim3 blockdim(32,32,1)....

反正最后相乘起来结果为1024.

至于grid：
硬件限制由 CUDA 运行时和 GPU 的硬件寄存器固定，具体为：

gridDim.x 最大值： 2³¹ - 1 = 2147483647

gridDim.y 最大值： 65535

gridDim.z 最大值： 65535

![A100 108个SM架构](image1.png)

### 1. Orin GPU结构
![Orin GPU规格](image2.png)

### 2. 大模型量化技术原理

模型压缩主要分为如下几类：

- 剪枝（Pruning）
- 知识蒸馏（Knowledge Distillation）
- 量化

大模型量化面临着下面这些问题：

将 LLM 进行低比特权重量化可以节省内存，但却很难实现。量化感知训练（QAT）由于训练成本较高并不实用，而训练后量化（PTQ）在低比特场景下面临较大的精度下降。


这里有一些基本的大模型量化技术原理的介绍：
[大模型量化技术原理](https://zhuanlan.zhihu.com/p/681578090)

#### 2.1 AWQ、AutoAWQ
**AWQ（AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration）是一种对大模型仅权重量化方法。通过保护更“重要”的权重不进行量化，从而在不进行训练的情况下提高准确率。**

原理是：权重对于LLM的性能并不同等重要”的观察，存在约（0.1%-1%）显著权重对大模型性能影响太大，通过跳过这1%的显著权重（salient weight）不进行量化，可以大大减少量化误差。但是这种方法由于是混合精度的，对硬件并不是很友好，这里AWQ技术主要是针对解决这个问题，所有权重都量化，但是是通过激活感知缩放保护显著权重。

**具体细节在上面文章链接当中查看**


![AWQ](image3.png)
#### 2.2 LLM.int8()
属于：RTN（round-to-nearest (RTN) 量化）

（论文：LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale）是一种采用混合精度分解的量化方法。该方案先做了一个矩阵分解，对绝大部分权重和激活用8bit量化（vector-wise）。对离群特征的几个维度保留16bit，对其做高精度的矩阵乘法。

![int8()](image4.jpg)

但是这种分离计算的方式拖慢了推理速度


int8量化是量化 任何模型最简单的方法之一，这里对于了离群值的处理回大大拖慢了推理速度，因为在计算过程中分成两部分计算，离群值部分用fp16来计算，被int8量化的非离群值用int8来计算。

#### 2.3 GPTQ
GPTQ(论文：GPTQ: ACCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PR E-TRAINED TRANSFORMERS) 采用 int4/fp16 (W4A16) 的混合量化方案，其中模型权重被量化为 int4 数值类型，而激活值则保留在 float16，是一种仅权重量化方法。在推理阶段，模型权重被动态地反量化回 float16 并在该数值类型下进行实际的运算；同 OBQ 一样，GPTQ还是从单层量化的角度考虑，希望找到一个量化过的权重，使的新的权重和老的权重之间输出的结果差别最小。

![GPTQ](image5.jpg)

那GPTQ有什么优势呢？


![GPTQ](image6.jpg)

