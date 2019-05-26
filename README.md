# 利用强化学习训练 Agent 玩 Atari 游戏
## 复现论文
### 神经网络结构
输入：84x84x4的图片  
第一层卷积：16个8x8卷积核，步长为4  
第二层卷积：32个4x4卷积核，步长为2  
第三层全连接：输入32x9x9，输出256  
第四层全连接：输入256，输出6对应6个动作  
（我们考虑在卷积与全连接之间加一层最大池化层以增强特征，但由于训练轮数限制暂不能比较出两者的差异）  
### 记忆池
通过队列或数组存储每一步的[state, action, reward, next_state, done]
### 行为决策：ϵ-greedy 策略
ϵ-greedy 策略兼具探索与利用的功能，在已知与未知之间进行了平衡。  
通过引入随机行为，增强了对游戏的探索，同时能产生更多有用的记忆。  
并且随着训练的进行逐渐减少随机行为，使最终的模型能利用已有的训练记忆更好地进行决策。
### 神经网络优化
根据论文算法，Q的更新公式为  
Q(s,a) ← Q(s,a)+α[r+γmaxa'Q(s',a')−Q(s,a)]   
损失函数为  
L(θ)=E[(TargetQ−Q(s,a;θ))^2]  
通过对loss的梯度下降调整神经网络参数

