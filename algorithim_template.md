# 这个md记录秋招刷题过程中遇到的一些算法板子写法
（先从快排写起吧！）
## 快速排序
```C++
// 比较简单的写法了，记住！！
    void quick_sort(vector<int> &nums,int left,int right){
        if(left>=right)return;
        int flag = nums[left];
        int i = left,j = right;
        while(i<j){
            while(i<j&&nums[j]>=flag)--j;
            while(i<j&&nums[i]<=flag)++i;
            if(i<j)swap(nums[i],nums[j]);
        }
        swap(nums[left],nums[i]);
        quick_sort(nums,left,i-1);
        quick_sort(nums,i+1,right);
    }
```