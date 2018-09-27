#! /usr/bin/python3

import numpy as np

class DispatchOrder():
	'''
	滴滴出行派单模型
	基于论文: A Taxi Order Dispatch Model based On Combinatorial Optimization
	本版本限制 N <= M, 即每轮派单都有足够可用的司机
	'''
	def __init__(self, N=5, M=5):
		'''
		初始化
		Args:
			N: 订单数
			M: 司机数
			N <= M
		'''
		self.N = N
		self.M = M
	
	def init_prob(self):
		'''
		初始化每个司机对不同订单的接受概率
		PS: 论文是基于数据集进行训练得到
		TO DO: 基于模型获得概率矩阵
		'''
		self.prob = np.random.rand(self.N, self.M)
		#self.prob = np.array([[0.45801735,0.33896261,0.09375837,0.00598978,0.74108335],
		#						[0.560608,0.4972837,0.25953291,0.70191309,0.83292756]])
	
	def init_dispatch_mat(self):
		'''
		初始化派单矩阵
		'''
		self.A = np.row_stack((np.ones((1,self.M), dtype=np.int), np.zeros((self.N-1,self.M), dtype=np.int)))
		#self.A = np.column_stack((np.eye((self.N), dtype=np.int), np.zeros((self.N, self.M-self.N), dtype=np.int)))
		# 用于优化算法
		self.B = self.A.transpose()
	
	def calc_sr(self, i):
		'''
		计算订单i的接单成功率
		Args:
			i: 第i个订单
		Returns:
			sr: 第i个订单的接单成功率
		'''
		unaccept_prob = 1
		for j in range(self.M):
			unaccept_prob *= (1-self.prob[i][j])**self.A[i][j]
		sr_of_i = 1 - unaccept_prob
		return sr_of_i
	
	def calc_avg_sr(self, sr):
		'''
		计算得到全局接单成功率
		Args:
			sr: 所有订单的接单成功率
		Returns:
			avg_sr: 全局平均接单成功率
		'''
		avg_sr = np.mean(sr)
		return avg_sr
	
	def _find_undispatch(self, i):
		'''
		找到未分配订单i的所有司机
		Args:
			i: 订单i
		Returns:
			U: 未分配订单i的所有司机
		'''
		U = [j for j,value in enumerate(self.A[i]) if value == 0]
		return U
	
	def hill_climbing(self):
		'''
		爬山算法实现
		PS: 论文是基于爬山算法进行求解,其实也可尝试其他启发式算法
		TO DO: 遗传算法, 模拟退火算法
		'''
		self.init_prob()
		self.init_dispatch_mat()
		# 计算得到当前平均接单率
		sr = []
		for i in range(self.N):
			sr_of_i = self.calc_sr(i)
			sr.append(sr_of_i)
		avg_sr = self.calc_avg_sr(sr)
		for epoch in range(100):
			pre_sr = avg_sr
			for i in range(self.N):
				# 找到未分配订单i的所有司机
				U = self._find_undispatch(i)
				for j in U:
					# 找到第j个司机被分配的订单
					k_list = [k for k,value in enumerate(self.B[j]) if value == 1]
					# 还未分配订单
					if len(k_list) == 0:
						self.A[i][j] = 1
						self.B[j][i] = 1
						sr[i], sr[k] = self.calc_sr(i), self.calc_sr(k)
						avg_sr = self.calc_avg_sr(sr)
					else:
						k = k_list[0]
						# 如果修改订单分派的司机,提升了接单成功率,修改派单矩阵
						pre_sr_of_i = sr[i]
						pre_sr_of_k = sr[k]
						self.A[i][j] = 1
						self.A[k][j] = 0
						sr[i], sr[k] = self.calc_sr(i), self.calc_sr(k)
						post_avg_sr = self.calc_avg_sr(sr)
						if post_avg_sr > avg_sr:
							avg_sr = post_avg_sr
							self.B[j][i] = 1
							self.B[j][k] = 0
						# 还原
						else:
							sr[i] = pre_sr_of_i
							sr[k] = pre_sr_of_k
							self.A[i][j] = 0
							self.A[k][j] = 1
						print('avg_sr: {:>6.3f}'.format(avg_sr))
						#print(self.A)
						#print(self.prob)
			if avg_sr - pre_sr <= 1e-8:
				break
				
if __name__ == '__main__':
	dispatchOrder = DispatchOrder()
	dispatchOrder.hill_climbing()
	print(dispatchOrder.A)
	print(dispatchOrder.prob)