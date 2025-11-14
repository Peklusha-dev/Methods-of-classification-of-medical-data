import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Загрузка данных ирисов Фишера
ds = load_iris()
data = ds.data
feat_names = ds.feature_names
target = ds.target
target_names = ds.target_names

# Создаем DataFrame
full_df = pd.DataFrame(data, columns=feat_names)
full_df['target'] = target
full_df['target_name'] = full_df['target'].map(lambda x: target_names[int(x)])

# 2. Визуализация ирисов в пространстве длины и ширины лепестка
plt.figure(figsize=(8, 6))

for target_name in target_names:
    subset = full_df[full_df['target_name'] == target_name]
    plt.scatter(
        subset[feat_names[2]],  # длина лепестка
        subset[feat_names[3]],  # ширина лепестка
        label=target_name,
        alpha=0.7
    )

plt.xlabel(feat_names[2])
plt.ylabel(feat_names[3])
plt.title('Классы ирисов в пространстве длины и ширины лепестка')
plt.legend()
plt.grid(True)
plt.show()

# 3. Создание двух групп двумерных данных
# Первая группа
group1_x = np.random.normal(loc=4, scale=1, size=10)
group1_y = np.random.normal(loc=6, scale=2, size=10)
group1 = np.column_stack([group1_x, group1_y])

# Вторая группа
group2_x = np.random.normal(loc=7, scale=3, size=10)
group2_y = np.random.normal(loc=2, scale=1, size=10)
group2 = np.column_stack([group2_x, group2_y])

# Визуализация двух групп
plt.figure(figsize=(6, 6))
plt.scatter(group1[:, 0], group1[:, 1], color='blue', label='Класс 1', alpha=0.7)
plt.scatter(group2[:, 0], group2[:, 1], color='red', label='Класс 2', alpha=0.7)
plt.xlabel('Первая компонента')
plt.ylabel('Вторая компонента')
plt.title('Две группы двумерных данных')
plt.legend()
plt.grid(True)
plt.show()