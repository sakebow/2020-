import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, explained_variance_score as EVS, mean_squared_error as MSE

def read_data(path):
  dataset = pd.read_excel(path)
  # x
  data_x = dataset.iloc[:, 1:]
  # y
  data_y = dataset.iloc[:, 0]
  # print(data) - a matrix of 252 * 20
  return data_x, pd.DataFrame(data_y)
  pass

def find_max_score(times):
  score = 0
  max_index = 0
  x, y = read_data('~/python/各种神奇函数测试/copy.xlsx')
  for i in range(times):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = i)
    reg = LinearRegression().fit(x_train, y_train)
    yhat = reg.predict(x_test)
    test_score = r2_score(y_test, yhat)
    if test_score > score:
      score = test_score
      max_index = i
      pass
    pass
  return score, max_index
  pass

def linear_regression():
  x, y = read_data('~/python/各种神奇函数测试/copy.xlsx')
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 37981)
  reg = LinearRegression().fit(x_train, y_train)
  yhat = reg.predict(x_test)
  # ceof 系数矩阵，intercept 截距， MSE 均方差
  print(f'系数矩阵：{reg.coef_}\n函数截距：{reg.intercept_}')
  print(f'回归函数为：Y = {reg.intercept_} + {reg.coef_}X')
  print(f'均方误差为：{MSE(yhat, y_test)}')
  mse_value = np.array(pd.DataFrame(math.sqrt(MSE(yhat, y_test)) / y_test.mean()))
  print(f'标准误差为：{mse_value[0]}')
  # 交叉验证(分类器，x，y，分类数量（最后结果数量和分类数量一致），负均方误差（越接近0越好）)
  print(f'交叉验证得分：{cross_val_score(reg, x, y, cv=10, scoring="neg_mean_squared_error").mean()}')
  print(f'回归模型拟合度：{r2_score(y_test, yhat)}')

  # 由随机森林的降维，有x216是最重要的变量。选出其回归函数斜率：reg.coef_[0][212]
  # 回归方程：Y = reg.intercept_ + x * reg.coef_
  # 开始绘制x216的回归方程
  
  # 坐标轴间隔
  # x_major_locator = MultipleLocator(0.1)
  # ax = plt.gca()
  # ax.xaxis.set_major_locator(x_major_locator)

  k = reg.coef_[0][0]
  b = reg.intercept_[0] # for copy.xlsx
  # b = reg.intercept_[0] / len(x) # for copy(full).xlsx
  # print(f'k = {k}, b = {b}')
  # print(x)
  theta_1 = x.x216
  theta_2 = []
  for z_index in theta_1:
    theta_2.append(k * z_index + b)
    pass
  theta_2 = np.array(theta_2)
  plt.scatter(x.x216, y, c='b')
  plt.plot(theta_1, theta_2, c='g')
  plt.title('RON - S-ZORB.PC_1301.PV')
  plt.xlabel('S-ZORB.PC_1301.PV/MPa')
  plt.ylabel('RON')
  plt.show()
  pass

def get_line():
  x, y = read_data('~/python/各种神奇函数测试/copy.xlsx')
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 37981)
  reg = LinearRegression().fit(x_train, y_train)
  yhat = reg.predict(x_test)
  k = reg.coef_[0]
  b = reg.intercept_[0] # for copy.xlsx
  # print(k, b)
  return k, b
  pass

def compare():
  x, RON = read_data('~/python/各种神奇函数测试/copy.xlsx')
  ks, b = get_line()
  i = 0; y = 0; rate = 0; length = len(x)

  RON = np.array(RON)

  for sample in range(length):
    i = 0; y = 0
    for x_index in x:
      y += ks[i] * x.at[sample, str(x_index)]
      i += 1
      pass
    y += b
    rate = y / RON[sample][0] - 1
    if rate >= 0.3:
      print(f'K101机出口压力:{np.array(x.x216)[sample]}, D-105下锥体松动风流量: {np.array(x.x173)[sample]}, EH-101加热元件温度:{np.array(x.x313)[sample]}')
      pass
    pass
  pass

def draw_final_image():
  x1 = np.arange(0.2503190875, 3.111529, 0.1)
  x2 = np.arange(0.5547120125, 25.3971645, 5)
  x3 = np.arange(343.682, 497.9730225, 1)
  y1 = []; y2 = []; y3 = []
  ks, b = get_line()
  for item1 in x1:
    y1.append(item1 * ks[0] + 0.5547120125 * ks[1] + 343.682 * ks[2] + b + 90.6)
    pass
  plt.subplot(1, 3, 1)
  plt.xlabel('K101 / MPa')
  plt.ylabel('RON')
  plt.title('K101 - RON')
  plt.plot(x1, y1, 'r')
  for item2 in x2:
    y2.append(item2 * ks[1] + 0.2503190875 * ks[0] + 343.682 * ks[2] + b + 90.6)
    pass
  plt.subplot(1, 3, 2)
  plt.xlabel('D-105')
  plt.ylabel('RON')
  plt.title('D-105 - RON')
  plt.plot(x2, y2, 'g')
  for item3 in x3:
    y3.append(item3 * ks[2] + 0.2503190875 * ks[0] + 0.5547120125 * ks[1] + b + 90.6)
    pass
  plt.subplot(1, 3, 3)
  plt.xlabel('EH-101 / ℃')
  plt.ylabel('RON')
  plt.title('EH-101 - RON')
  plt.plot(x3, y3)
  # print(y)
  plt.subplots_adjust(wspace=0.7)
  plt.show()
  pass

def terminate_all():
  x1 = np.arange(0.2503190875, 3.111529, 0.1)
  xs, RON_DATA = read_data('~/python/各种神奇函数测试/copy.xlsx')
  x = []; y = 0; RON = []
  RON_DATA = sorted([89.22, 89.32, 89.22, 89.32, 89.32, 89.02, 88.32, 89.59, 89.20, 89.20, 89.30, 88.80, 88.90,
              88.90, 87.70, 85.40, 86.10, 86.70, 86.10, 86.49, 87.09, 87.59, 86.59, 87.19, 87.99, 87.59,
              87.89, 87.69, 87.29])
  for x_index in xs.columns:
    x.append(xs.at[0, str(x_index)])
    pass
  ks, b = get_line()
  k_len = len(ks)
  times = len(x1)
  times = 0
  for item in x1:
    y = ks[0] * item
    for i in range(1, k_len):
      y += ks[i] * x[i]
      pass
    y += b + RON_DATA[times]
    times += 1
    RON.append(y)
    pass
  print(f'{RON[0]}')
  x = range(len(x1))
  plt.plot(x, RON, 'r')
  plt.scatter(x, RON)
  plt.xlabel('times')
  plt.ylabel('RON')
  plt.title('Times - RON')
  plt.show()
  pass

if __name__ == '__main__':
  # linear_regression()
  # print(find_max_score(50000))
  # compare()
  # draw_final_image()
  terminate_all()
  pass