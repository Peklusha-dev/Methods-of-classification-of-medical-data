import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Ввод матриц данных f1 и f2 для двух классов A и B
f1 = np.array(
	[
		[0.997, 0.05, 0.15],
		[0.997, 0.09, 0.28],
		[1.221, 0.09, 0.40],
		[0.997, 0.10, 0.39],
		[0.997, 0.10, 0.40],
		[1.709, 0.13, 0.72],
		[1.953, 0.22, 0.91],
		[1.221, 0.07, 0.28],
		[2.441, 0.17, 0.76],
		[2.686, 0.19, 0.80],
		[2.930, 0.15, 0.54],
		[2.930, 0.13, 0.72],
		[1.221, 0.11, 0.56],
		[1.953, 0.19, 0.72],
		[0.997, 0.13, 0.53],
		[1.221, 0.12, 0.41],
		[1.221, 0.07, 0.29],
		[0.997, 0.11, 0.51],
	]
)

f2 = np.array(
	[
		[3.174, 0.17, 0.63],
		[2.686, 0.19, 0.71],
		[3.174, 0.10, 0.42],
		[2.686, 0.19, 0.73],
		[2.686, 0.21, 0.77],
		[2.441, 0.22, 0.87],
		[2.930, 0.23, 0.83],
		[2.930, 0.23, 0.85],
		[3.418, 0.22, 0.84],
		[3.174, 0.20, 0.84],
		[3.418, 0.22, 0.80],
		[3.662, 0.24, 0.83],
		[3.662, 0.22, 0.85],
		[3.906, 0.22, 0.81],
		[3.906, 0.22, 0.85],
		[3.906, 0.24, 0.88],
		[3.906, 0.24, 0.84],
		[3.906, 0.22, 0.86],
		[3.906, 0.23, 0.85],
		[3.906, 0.23, 0.81],
		[4.150, 0.22, 0.87],
		[3.906, 0.11, 0.61],
	]
)

# 2. Вычисление переменной группировки g
g = np.zeros(f1.shape[0])
g[:f1.shape[0]] = 1  # Класс A (группа 1)
g = np.append(g, np.zeros(f2.shape[0]))
g[f1.shape[0]:] = 2  # Класс B (группа 2)

print("Переменная группировки g:")
print(g)

# 3. Вычисление главных компонент и визуализация
# Объединение матриц данных
X = np.vstack([f1, f2])

# Вычисление главных компонент
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Визуализация с использованием gscatter (аналог в matplotlib)
plt.figure(figsize=(10, 8))

# Создание scatter plot с цветовой кодировкой по группам
colors = ['red', 'blue']
labels = ['Class A', 'Class B']

for i, group in enumerate(np.unique(g)):
	mask = g == group
	plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
	            c=colors[i], label=labels[i], alpha=0.7, s=60)

plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA: Projection of Classes A and B')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Дополнительная визуализация с использованием seaborn для лучшего отображения групп
try:
	import seaborn as sns
	import pandas as pd

	# Создание DataFrame для удобства визуализации
	df_pca = pd.DataFrame({
		'PC1': X_pca[:, 0],
		'PC2': X_pca[:, 1],
		'Class': ['A' if val == 1 else 'B' for val in g]
	})

	plt.figure(figsize=(10, 8))
	sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Class',
	                palette={'A': 'red', 'B': 'blue'}, s=80, alpha=0.8)
	plt.title('PCA: Class Separation Visualization')
	plt.grid(True, alpha=0.3)
	plt.show()

except ImportError:
	print("Seaborn не установлен. Используется стандартная matplotlib визуализация.")

# 4. Анализ результатов
print("\n" + "=" * 50)
print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
print("=" * 50)

# Информация о главных компонентах
print(f"Объясненная дисперсия компонент: {pca.explained_variance_ratio_}")
print(f"Суммарная объясненная дисперсия: {sum(pca.explained_variance_ratio_):.2%}")

# Анализ разделимости классов
# Простая оценка разделимости по проекциям на PC1
class_a_pc1 = X_pca[g == 1, 0]
class_b_pc1 = X_pca[g == 2, 0]

print(f"\nСтатистика по PC1:")
print(f"Класс A: среднее = {class_a_pc1.mean():.3f}, std = {class_a_pc1.std():.3f}")
print(f"Класс B: среднее = {class_b_pc1.mean():.3f}, std = {class_b_pc1.std():.3f}")

# Оценка перекрытия классов
overlap_threshold = (class_a_pc1.max() + class_b_pc1.min()) / 2
print(f"\nГраница разделения по PC1: {overlap_threshold:.3f}")

# Подсчет потенциальных ошибок классификации
misclassified_a = np.sum(class_a_pc1 > overlap_threshold)
misclassified_b = np.sum(class_b_pc1 < overlap_threshold)

total_samples = len(g)
error_rate = (misclassified_a + misclassified_b) / total_samples

print(f"\nОЦЕНКА ОШИБОК КЛАССИФИКАЦИИ:")
print(f"Неверно классифицировано из класса A: {misclassified_a}/{len(class_a_pc1)}")
print(f"Неверно классифицировано из класса B: {misclassified_b}/{len(class_b_pc1)}")
print(f"Общая оценка ошибки: {error_rate:.2%}")

# ВЫВОД
print(f"\nВЫВОД:")
if error_rate < 0.1:
	print("Классы хорошо разделимы в пространстве главных компонент")
	print("Возможность линейного разделения: ВЫСОКАЯ")
elif error_rate < 0.2:
	print("Классы умеренно разделимы в пространстве главных компонент")
	print("Возможность линейного разделения: СРЕДНЯЯ")
else:
	print("Классы плохо разделимы в пространстве главных компонент")
	print("Возможность линейного разделения: НИЗКАЯ")

print(f"Рекомендуется использовать {sum(pca.explained_variance_ratio_):.1%} информации для классификации")