# This is parameter setting for all deep learning algorithms
import sys
# Import games
sys.path.append("DQN_GAMES/")

# Action Num  游戏种类
import pong
import dot
import dot_test
import tetris
import wormy
import breakout as game

Gamma = 0.99           #未来回报折扣衰减系数
Learning_rate = 0.00025#神经网络参数学习率
Epsilon = 1            #深度强化学习的小概率随机搜索策略初始概率大小
Final_epsilon = 0.1    #深度强化学习的小概率随机搜索策略最终概率大小

Num_action = game.Return_Num_Action()#游戏可执行的动作数目

Num_replay_memory = 50000  #DQN的经验回放池容量大小
Num_start_training = 50000 #DQN训练开始前，观察次数
Num_training = 500000  #DQN训练次数
Num_update = 5000      #深度强化学习网络模型保存频率
Num_batch = 32         #DQN每次训练每批次数为32
Num_test = 250000      #DQN测试次数
Num_skipFrame = 4      #深度强化学习跳帧
Num_stackFrame = 4     #DQN保存帧数
Num_colorChannel = 1   #数据通道

Num_plot_episode = 50  #每训练50次计算显示一次损失函数

GPU_fraction = 0.2
Is_train = True     #是否为训练状态
Load_path = ''      #神经网络模型保存路径   这里为工程根目录

img_size = 80       #输入数据大小

first_conv   = [8,8,Num_colorChannel * Num_stackFrame,32] #第1层卷积核大小
second_conv  = [4,4,32,64]        #第2层卷积核大小
third_conv   = [3,3,64,64]        #第3层卷积核大小
first_dense  = [10*10*64, 512]    #第1层全连接层大小
second_dense = [512, Num_action]  #第2层全连接层大小
