# Cuda 编程相关

- nvprof在Ampere系列GPU已经不再适用了，用nsys(Nsight Systems)
- \_\_shared\_\_ shared memery 的使用方法：
  1. 从软件概念来讲，一个共享内存是在一个block内，block内的所有线程都能使用该共享内存
  2. 从硬件概念来讲，一个共享内存是在一个sm中
  3. 在核函数当中声明的shared memery 在整个block当中可以共享，跟随整个block的生命周期
- 核函数是针对每个线程进行编程的
- 共享内存在每个线程块内内存共享，线程块之间不共享
- cutlass，cublas，cudnn的区别和联系

![alt text](image1.png)

- flash attention中保存的两个中间变量m 和l ，l是为了求sfotmax的求和，来当分母，m是做softmax时防止数值过大所以要减去的局部最大值max
- 分析cuda性能常见的性能指标：
  1. 占用率（Occupancy）：活跃warp与最大可能warp的比率
  2. 内存带宽利用率：实际带宽与理论峰值带宽的比率
  3. 分支发散：warp内线程执行不同路径的情况
  4. 内存访问模式：合并访问vs非合并访问
  5. 共享内存bank冲突
- 性能分析工具：Nsight Systems(nsys),Nsight Compute (ncu)
