# Deep Q-Network Algorithm

# Import modules
import tensorflow as tf
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import os

# Import game
import sys
sys.path.append("DQN_GAMES/")
'''当导入模块时：import xxx，默认情况下python解析器会搜索当前目录、已安装的内置模块和第三方模块，
搜索路径存放在sys模块的path中,当我们要添加自己的搜索目录时，可以通过列表的append()方法；对于模块
和自己写的脚本不在同一个目录下，在脚本开头加sys.path.append('xxx')'''
import Deep_Parameters        #深度神经网络参数设置
game = Deep_Parameters.game   #获取游戏终端状态

class DQN:
	def __init__(self):

		# Game Information
		self.algorithm = 'DQN'             #深度强化学习使用算法
		self.game_name = game.ReturnName() #游戏名

		# Get parameters
		self.progress = ''
		self.Num_action = game.Return_Num_Action()#游戏可执行的动作数目

		# Initial parameters
		self.Num_Exploration = Deep_Parameters.Num_start_training #DQN训练开始前，观察次数
		self.Num_Training    = Deep_Parameters.Num_training       #DQN训练次数
		self.Num_Testing     = Deep_Parameters.Num_test           #网络测试次数

		self.learning_rate = Deep_Parameters.Learning_rate  #参数学习率
		self.gamma = Deep_Parameters.Gamma                  #为折扣衰减系数

		self.first_epsilon = Deep_Parameters.Epsilon       #深度强化学习的小概率随机搜索策略初始概率大小
		self.final_epsilon = Deep_Parameters.Final_epsilon #深度强化学习的小概率随机搜索策略最终概率大小

		self.epsilon = self.first_epsilon   #深度强化学习的小概率随机搜索策略初始概率大小

		self.Num_plot_episode = Deep_Parameters.Num_plot_episode #每训练50次计算显示一次损失函数

		self.Is_train = Deep_Parameters.Is_train    #是否为训练状态
		self.load_path = Deep_Parameters.Load_path  #网络模型加载

		self.step = 1
		self.score = 0
		self.episode = 0

		# date - hour - minute of training time #训练时间显示
		self.date_time = str(datetime.date.today()) + '_' + \
		                 str(datetime.datetime.now().hour) + '_' + \
						 str(datetime.datetime.now().minute)

		# parameters for skipping and stacking
		self.state_set = []
		self.Num_skipping = Deep_Parameters.Num_skipFrame    #每间隔多少帧执行一次动作
		self.Num_stacking = Deep_Parameters.Num_stackFrame   #每间隔多少帧保存一次状态序列

		# Parameter for Experience Replay
		self.Num_replay_memory = Deep_Parameters.Num_replay_memory #DQN的经验回放池容量大小
		self.Num_batch = Deep_Parameters.Num_batch                 #DQN每次训练每批次数为32
		self.replay_memory = []          #初始化经验回放池

		# Parameter for Target Network
		self.Num_update_target = Deep_Parameters.Num_update

		# Parameters for network
		self.img_size = 80
		self.Num_colorChannel = Deep_Parameters.Num_colorChannel
                #卷积神经网络模型架构参数设置
		self.first_conv   = Deep_Parameters.first_conv   #第一层卷积
		self.second_conv  = Deep_Parameters.second_conv  #第二层卷积
		self.third_conv   = Deep_Parameters.third_conv   #第三层卷积
		self.first_dense  = Deep_Parameters.first_dense  #第一层全连接层
		self.second_dense = Deep_Parameters.second_dense #第二层全连接层

		self.GPU_fraction = Deep_Parameters.GPU_fraction   #网络模型训练分配GPU计算能力

		# Variables for tensorboard
		self.loss = 0        #网络模型损失函数变化
		self.maxQ = 0        #最大Q值变化
		self.score_board = 0 #游戏及时回报
		self.maxQ_board  = 0 #游戏最大Q值
		self.loss_board  = 0 #网络模型损失函数
		self.step_old    = 0 #网络训练次数记录

		# Initialize Network
		self.input, self.output = self.network('network')              #获取网络输入输出数据格式
		self.input_target, self.output_target = self.network('target') #获取网络输入输出数据格式
		self.train_step, self.action_target, self.y_target, self.loss_train = self.loss_and_train()
		self.sess, self.saver, self.summary_placeholders, self.update_ops, self.summary_op, self.summary_writer = self.init_sess()

	def main(self):
		# Define game state
		game_state = game.GameState()#获取游戏状态信息

		# Initialization
		state = self.initialization(game_state)#获取初始游戏状态
		stacked_state = self.skip_and_stack_frame(state)#

		while True:
			# Get progress:   #获取当前处于训练、测试那个阶段
			self.progress = self.get_progress()

			# Select action   根据当前状态选择动作
			action = self.select_action(stacked_state)

			# Take action and get info. for update选择动作，接收信息
			next_state, reward, terminal = game_state.frame_step(action)#游戏终端执行动作，返回及时回报和下一状态以及终端状态
			next_state = self.reshape_input(next_state)#数据预处理
			stacked_next_state = self.skip_and_stack_frame(next_state)#跳帧存储状态

			# Experience Replay经验回放池存储状态转移序列
			self.experience_replay(stacked_state, action, reward, stacked_next_state, terminal)

			# Training!训练
			if self.progress == 'Training':
				# Update target network更新目标网络
				if self.step % self.Num_update_target == 0:
					self.update_target()

				# Training网络训练
				self.train(self.replay_memory)

				# Save model保存网络模型
				self.save_model()

			# Update former info.
			stacked_state = stacked_next_state
			self.score += reward#累积回报
			self.step += 1

			# Plotting   #画出tensorboard
			self.plotting(terminal)

			# If game is over (terminal)
			if terminal:
				stacked_state = self.if_terminal(game_state)#终端状态

			# Finished!  self.progress表示模型处于观察状态、搜索状态、训练状态  
			if self.progress == 'Finished':
				print('Finished!')
				break

	def init_sess(self):
		# Initialize variables
		config = tf.ConfigProto()#用在创建session的时候。用来对session进行参数配置
		config.gpu_options.per_process_gpu_memory_fraction = self.GPU_fraction#占用 self.GPU_fraction 显存 

		sess = tf.InteractiveSession(config=config)

		# Make folder for save data
		os.makedirs('saved_networks/' + self.game_name + '/' + self.date_time + '_' + self.algorithm)

		# Summary for tensorboard  tensorflow的可视化tensorboard
		summary_placeholders, update_ops, summary_op = self.setup_summary()
		summary_writer = tf.summary.FileWriter('saved_networks/' + self.game_name + '/' + self.date_time + '_' + self.algorithm, sess.graph)

		init = tf.global_variables_initializer()
		sess.run(init)

		# Load the file if the saved file exists训练网络后想保存训练好的模型，以及在程序中读取以保存的训练好的模型
		# 首先，保存和恢复都需要实例化一个 tf.train.Saver
		saver = tf.train.Saver()
		# check_save = 1  是否加载已训练好的模型
		check_save = input('Load Model? (1=yes/2=no): ')

		if check_save == 1:
			# Restore variables from disk.使用 saver.restore() 方法，重载模型的参数，继续训练或用于测试数据
			saver.restore(sess, self.load_path + "/model.ckpt")  #神经网络模型保存位置
			print("Model restored.")

			check_train = input('Inference or Training? (1=Inference / 2=Training): ')#用来获取控制台的输入数据，返回为 string 类型。
			if check_train == 1:
				self.Num_Exploration = 0   #停止策略搜索
				self.Num_Training = 0      #停止网络训练
		return sess, saver, summary_placeholders, update_ops, summary_op, summary_writer

	def initialization(self, game_state):
		action = np.zeros([self.Num_action])        #定义动作数组
		state, _, _ = game_state.frame_step(action) #对终端输入动作   获取相应的状态回报和下一个状态
		state = self.reshape_input(state)           #对状态数据进行预处理

		for i in range(self.Num_skipping * self.Num_stacking):
			self.state_set.append(state)

		return state
        #动态跳帧
	def skip_and_stack_frame(self, state):
		self.state_set.append(state)

		state_in = np.zeros((self.img_size, self.img_size, self.Num_colorChannel * self.Num_stacking))

		# Stack the frame according to the number of skipping frame
		for stack_frame in range(self.Num_stacking):
			state_in[:,:,stack_frame] = self.state_set[-1 - (self.Num_skipping * stack_frame)]

		del self.state_set[0]

		state_in = np.uint8(state_in)
		return state_in

        #获取网络模型阶段
	def get_progress(self):
		progress = ''
		if self.step <= self.Num_Exploration:
			progress = 'Exploring'#策略搜索状态
		elif self.step <= self.Num_Exploration + self.Num_Training:
			progress = 'Training' #网络训练状态
		elif self.step <= self.Num_Exploration + self.Num_Training + self.Num_Testing:
			progress = 'Testing'  #网络测试状态
		else:
			progress = 'Finished' 

		return progress

        #图像数据预处理裁剪灰度值化
	# Resize and make input as grayscale  
	def reshape_input(self, state):
		state_out = cv2.resize(state, (self.img_size, self.img_size))  #数据resize为网络大小
		if self.Num_colorChannel == 1:
			state_out = cv2.cvtColor(state_out, cv2.COLOR_BGR2GRAY)#数据由RGB变为灰度图像
			state_out = np.reshape(state_out, (self.img_size, self.img_size))

		state_out = np.uint8(state_out)#变为单通到图像

		return state_out

	# Code for tensorboard
	def setup_summary(self):
	    episode_score = tf.Variable(0.)
	    episode_maxQ = tf.Variable(0.)
	    episode_loss = tf.Variable(0.)

	    tf.summary.scalar('Average Score/' + str(self.Num_plot_episode) + ' episodes', episode_score)
	    tf.summary.scalar('Average MaxQ/' + str(self.Num_plot_episode) + ' episodes', episode_maxQ)
	    tf.summary.scalar('Average Loss/' + str(self.Num_plot_episode) + ' episodes', episode_loss)

	    summary_vars = [episode_score, episode_maxQ, episode_loss]

	    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
	    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
	    summary_op = tf.summary.merge_all()
	    return summary_placeholders, update_ops, summary_op

	# Convolution and pooling
	def conv2d(self, x, w, stride):
		return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

	# Get Variables
	def conv_weight_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())

	def weight_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

	def bias_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

	def network(self, network_name):
		# Input   网络模型输入数据维定义
		x_image = tf.placeholder(tf.float32, shape = [None,self.img_size,self.img_size,self.Num_stacking * self.Num_colorChannel])
		x_normalize = (x_image - (255.0/2)) / (255.0/2)

		with tf.variable_scope(network_name):
			# Convolution variables
			w_conv1 = self.conv_weight_variable(network_name + '_w_conv1', self.first_conv)
			b_conv1 = self.bias_variable(network_name + '_b_conv1',[self.first_conv[3]])

			w_conv2 = self.conv_weight_variable(network_name + 'w_conv2',self.second_conv)
			b_conv2 = self.bias_variable(network_name + '_b_conv2',[self.second_conv[3]])

			w_conv3 = self.conv_weight_variable(network_name + '_w_conv3',self.third_conv)
			b_conv3 = self.bias_variable(network_name + '_b_conv3',[self.third_conv[3]])

			# Densely connect layer variables
			w_fc1 = self.weight_variable(network_name + 'w_fc1',self.first_dense)
			b_fc1 = self.bias_variable(network_name + 'b_fc1',[self.first_dense[1]])

			w_fc2 = self.weight_variable(network_name + 'w_fc2',self.second_dense)
			b_fc2 = self.bias_variable(network_name + 'b_fc2',[self.second_dense[1]])

		# Network   网络模型定义
		h_conv1 = tf.nn.relu(self.conv2d(x_normalize, w_conv1, 4) + b_conv1)
		h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
		h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)

		h_pool3_flat = tf.reshape(h_conv3, [-1, self.first_dense[0]])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1)+b_fc1)

		output = tf.matmul(h_fc1, w_fc2) + b_fc2
		return x_image, output

        #网络模型损失函数定义
	def loss_and_train(self):
		# Loss function and Train
		action_target = tf.placeholder(tf.float32, shape = [None, self.Num_action])
		y_target = tf.placeholder(tf.float32, shape = [None])#当前状态的Q值

		y_prediction = tf.reduce_sum(tf.multiply(self.output, action_target), reduction_indices = 1)#当前状态的及时回报加下一状态回报对应最大Q值，即网络的目标Q值
		Loss = tf.reduce_mean(tf.square(y_prediction - y_target))#损失函数=网络的目标Q值-当前状态的Q值
		train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-02).minimize(Loss)#网络模型优化函数

		return train_step, action_target, y_target, Loss

        #选择动作
	def select_action(self, stacked_state):
		action = np.zeros([self.Num_action])
		action_index = 0

		# Choose action
		if self.progress == 'Exploring':#搜索策略状态
			# Choose random action
			action_index = random.randint(0, self.Num_action-1)#随机动作
			action[action_index] = 1

		elif self.progress == 'Training':#网络模型处于训练状态
			if random.random() < self.epsilon:#采用小概率随机策略
				# Choose random action
				action_index = random.randint(0, self.Num_action-1)#选择随机动作
				action[action_index] = 1
			else:#采用贪婪策略（网络模型输入数据为状态s,输出多个动作所对应的Q值）
				# Choose greedy action   采用贪婪策略选择最大动作
				Q_value = self.output.eval(feed_dict={self.input: [stacked_state]})#网络模型输入动作，输出多个Q值
				action_index = np.argmax(Q_value)#选择最大Q值对应的动作
				action[action_index] = 1
				self.maxQ = np.max(Q_value)

			# Decrease epsilon while training  #对epsilon值进行更新，epsilon值表示多大概率进行随机搜索
			if self.epsilon > self.final_epsilon:#当前epsilon值大于停止迭代epsilon值则更新epsilon
				self.epsilon -= self.first_epsilon/self.Num_Training #每次epsilon减去初始epsilon除以模型训练次数

		elif self.progress == 'Testing':#网络模型处于测试阶段
			# Choose greedy action
			Q_value = self.output.eval(feed_dict={self.input: [stacked_state]})#网络模型输入动作输出各个动作的Q值
			action_index = np.argmax(Q_value)  #选择最大Q值
			action[action_index] = 1           #最大Q值对应的动作索引
			self.maxQ = np.max(Q_value)

			self.epsilon = 0   #小概率随机搜索为0

		return action

        #经验回放池
	def experience_replay(self, state, action, reward, next_state, terminal):  #更新经验回放池数据
		# If Replay memory is longer than Num_replay_memory, delete the oldest one
		if len(self.replay_memory) >= self.Num_replay_memory:   #经验回放池数据溢出，更新经验回放池，删除第一个
			del self.replay_memory[0]

		self.replay_memory.append([state, action, reward, next_state, terminal])  #添加新的状态序列转换

	def update_target(self):   #对于双网络模型  用当前网络参数更新目标网络参数
		# Get trainable variables
		trainable_variables = tf.trainable_variables()  #返回的是需要训练的变量列表
		# network variables   DDQN分当前网络
		trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

		# target variables    DDQN分目标网络
		trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

		for i in range(len(trainable_variables_network)):
			self.sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

	def train(self, replay_memory):
		# Select minibatch  随机抽取Num_batch数量的状态序列转换进行训练
		minibatch =  random.sample(replay_memory, self.Num_batch)

		# Save the each batch data
		state_batch      = [batch[0] for batch in minibatch] #状态s
		action_batch     = [batch[1] for batch in minibatch] #状态s执行动作a 
		reward_batch     = [batch[2] for batch in minibatch] #状态s执行动作a对应的及时回报r
		next_state_batch = [batch[3] for batch in minibatch] #状态s执行动作a对应的下一个状态
		terminal_batch   = [batch[4] for batch in minibatch] #终端状态是否为结束

		# Get y_prediction
		y_batch = []
		Q_batch = self.output_target.eval(feed_dict = {self.input_target: next_state_batch})

		# Get target values   状态动作值Q(s,a)的更新
		for i in range(len(minibatch)):
			if terminal_batch[i] == True:#如果终端结束   则及时回报就赋值给Q(s,a)
				y_batch.append(reward_batch[i])
			else:#如果终端非结束状态   则Q(s,a)为最大累积回报
				y_batch.append(reward_batch[i] + self.gamma * np.max(Q_batch[i]))

		_, self.loss = self.sess.run([self.train_step, self.loss_train], feed_dict = {self.action_target: action_batch,self.y_target: y_batch,self.input: state_batch})

	def save_model(self):
		# Save the variables to disk.在训练循环中，定期调用 saver.save() 方法，向文件夹中写入包含了当前模型中所有可训练变量的 checkpoint 文件
		if self.step == self.Num_Exploration + self.Num_Training:
		    save_path = self.saver.save(self.sess, 'saved_networks/' + self.game_name + '/' + self.date_time + '_' + self.algorithm + "/model.ckpt")
		    print("Model saved in file: %s" % save_path)

	def plotting(self, terminal):#画出tensorboard
		if self.progress != 'Exploring':#搜索阶段
			if terminal:
				self.score_board += self.score#累加得分，只记终端结束的得分

			self.maxQ_board  += self.maxQ#累加Q值，为下面计算平均
			self.loss_board  += self.loss#累加损失，为下面计算平均

			if self.episode % self.Num_plot_episode == 0 and self.episode != 0 and terminal:#训练了self.Num_plot_episode个数，且episode不为0（即网络在训练阶段），同时终端为结束状态
				diff_step = self.step - self.step_old      #训练了多少步
				tensorboard_info = [self.score_board / self.Num_plot_episode, self.maxQ_board / diff_step, self.loss_board / diff_step]#计算在过去训练diff_step步的平均得分，平均最大Q值，平均损失

				for i in range(len(tensorboard_info)):
				    self.sess.run(self.update_ops[i], feed_dict = {self.summary_placeholders[i]: float(tensorboard_info[i])})
				summary_str = self.sess.run(self.summary_op)
				self.summary_writer.add_summary(summary_str, self.step)

				self.score_board = 0  #重新累积分数
				self.maxQ_board  = 0  #重新累积Q值
				self.loss_board  = 0  #重新累积损失函数
				self.step_old = self.step  #重新计算次数
		else:
			self.step_old = self.step

	def if_terminal(self, game_state):            #终端状态
		# Show Progress     显示模型状态阶段
		print('Step: ' + str(self.step) + ' / ' +
		      'Episode: ' + str(self.episode) + ' / ' +
			  'Progress: ' + self.progress + ' / ' +
			  'Epsilon: ' + str(self.epsilon) + ' / ' +
			  'Score: ' + str(self.score))

		if self.progress != 'Exploring':
			self.episode += 1      #小概率值episode只在训练阶段起作用
		self.score = 0

		# If game is finished, initialize the state
		state = self.initialization(game_state)
		stacked_state = self.skip_and_stack_frame(state)  #动态跳帧

		return stacked_state

#__name__ 是内置变量，用于表示当前模块的名字。或者说表示当前的文件的名字
#if __name__ == '__main__':
#该句意思是如果模块是被直接运行的，则代码块被运行，如果模块是被导入的，则代码块不被运行。
#或者说运行的文件是当前的文件，则运行一下的代码；如果当前文件是被别的文件import的话，则别的文件运行时，不运行该句以下的代码，该句以下的代码只属于该文件运行时运行

if __name__ == '__main__':
	agent = DQN()
	agent.main()
