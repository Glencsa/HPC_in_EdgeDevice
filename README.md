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

### Orin GPU结构
![Orin GPU规格](image2.png)