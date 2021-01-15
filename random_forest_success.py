import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

datafile = '~/python/各种神奇函数测试/copy(full).xlsx'
data = pd.read_excel(datafile)
data_fea = data.iloc[:, 1:]  # 取数据中指标所在的列
# 准确率
model = RandomForestRegressor(random_state=100, max_depth=100, oob_score=True)
print(f'袋外数据测试得分：{model.fit(data_fea, data.y).oob_score_}')

model_feature_importance = model.feature_importances_

data_fea = data_fea.fillna(0)  # 随机森林只接受数字输入，不接受空值、逻辑值、文字等类型
data_fea = pd.get_dummies(data_fea)
model.fit(data_fea, data.y)

# 根据特征的重要性绘制柱状图
features = data_fea.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-25:]

# 重要性排序结果输出
# print(importances[indices])
# indices = np.argsort(importances[:35])
# 输出x_n和值
# for item in indices[::-1]:
#   if item < 8:
#     print(f'x_{item + 1}: {importances[item]}')
#     pass
#   else:
#     print(f'x_{item + 4}: {importances[item]}')
#     pass
#   pass

# 画重要性柱状图
plt.title('Importance Sort (DESC)')
rects = plt.barh(range(len(indices)), importances[indices], color='pink', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importance Value')
plt.ylabel('Index of x')
for rect in rects:
  # 如果柱比较短
  plt.text(rect.get_width(), rect.get_y() , f'{rect.get_width()}', color='red')
  rect.set_edgecolor('black')
  pass
plt.show()