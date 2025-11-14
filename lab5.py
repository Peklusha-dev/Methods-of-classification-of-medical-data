import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Загрузка и подготовка данных
ds = load_iris()
data = ds.data
feat_names = ds.feature_names
target = ds.target
target_names = ds.target_names

# Создаем DataFrame
full_df = pd.DataFrame(data, columns=feat_names)
full_df['target'] = target
full_df['target_name'] = full_df['target'].apply(lambda x: target_names[int(x)])

# Выбираем два класса для бинарной классификации (setosa vs virginica)
class1 = "setosa"
class2 = "virginica"

arr1 = full_df[full_df['target_name'] == class1][feat_names].values
arr2 = full_df[full_df['target_name'] == class2][feat_names].values

print(f"Размеры классов: {class1}: {arr1.shape}, {class2}: {arr2.shape}")

# 1. Вычисление весового вектора по Фишеру
m1 = arr1.mean(axis=0)
m2 = arr2.mean(axis=0)

# Внутриклассовые матрицы рассеяния
S1 = np.cov(arr1, rowvar=False) * (arr1.shape[0] - 1)
S2 = np.cov(arr2, rowvar=False) * (arr2.shape[0] - 1)
Sw = S1 + S2  # Общая внутриклассовая матрица рассеяния

# Весовой вектор w = Sw^(-1) * (m1 - m2)
w = np.linalg.pinv(Sw) @ (m1 - m2)
# Нормализуем вектор для устойчивости
w = w / np.linalg.norm(w)

print(f"Весовой вектор w: {w}")

# 2. Проекция данных на весовой вектор
proj1 = arr1 @ w
proj2 = arr2 @ w

print(f"\nПроекции {class1}: среднее = {proj1.mean():.3f}, std = {proj1.std():.3f}")
print(f"Проекции {class2}: среднее = {proj2.mean():.3f}, std = {proj2.std():.3f}")

# 3. Построение гистограмм
plt.figure(figsize=(10, 6))

# Определяем общие границы для гистограмм
all_projections = np.concatenate([proj1, proj2])
bins = np.linspace(all_projections.min(), all_projections.max(), 30)

plt.hist(proj1, bins=bins, alpha=0.7, label=class1, color='blue', density=True)
plt.hist(proj2, bins=bins, alpha=0.7, label=class2, color='red', density=True)

plt.xlabel('Значение проекции на весовой вектор')
plt.ylabel('Плотность вероятности')
plt.title('Гистораммы проекций классов на вектор Фишера')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# 4. Поиск оптимального порога
def find_optimal_threshold(proj1, proj2):
	"""Находит оптимальный порог как среднее между средними значениями проекций"""
	mean1, mean2 = np.mean(proj1), np.mean(proj2)
	optimal_threshold = (mean1 + mean2) / 2

	# Подсчет ошибок для этого порога
	# Определяем какое распределение левее
	if mean1 < mean2:
		error1 = np.sum(proj1 >= optimal_threshold)  # Class1 ошибочно справа
		error2 = np.sum(proj2 < optimal_threshold)  # Class2 ошибочно слева
	else:
		error1 = np.sum(proj1 < optimal_threshold)  # Class1 ошибочно слева
		error2 = np.sum(proj2 >= optimal_threshold)  # Class2 ошибочно справа

	total_error = error1 + error2
	error_rate = total_error / (len(proj1) + len(proj2))

	return optimal_threshold, (error1, error2, error_rate)


# Находим оптимальный порог
threshold, errors = find_optimal_threshold(proj1, proj2)
error1, error2, error_rate = errors

print(f"\n--- РЕЗУЛЬТАТЫ ---")
print(f"Оптимальный порог: {threshold:.4f}")
print(f"Ошибки классификации:")
print(f"  - {class1} ошибочно отнесены к {class2}: {error1}/{len(proj1)} ({error1 / len(proj1) * 100:.1f}%)")
print(f"  - {class2} ошибочно отнесены к {class1}: {error2}/{len(proj2)} ({error2 / len(proj2) * 100:.1f}%)")
print(f"  - Общая ошибка: {error1 + error2}/{len(proj1) + len(proj2)} ({error_rate * 100:.1f}%)")

# 5. Визуализация с порогом
plt.figure(figsize=(12, 6))

# Гистограммы
plt.hist(proj1, bins=bins, alpha=0.7, label=class1, color='blue', density=True)
plt.hist(proj2, bins=bins, alpha=0.7, label=class2, color='red', density=True)

# Вертикальная линия - оптимальный порог
plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
            label=f'Оптимальный порог: {threshold:.3f}')

plt.xlabel('Значение проекции на весовой вектор')
plt.ylabel('Плотность вероятности')
plt.title('Гистораммы проекций с оптимальным порогом классификации')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. Формулировка алгоритма классификации
print(f"\n--- АЛГОРИТМ КЛАССИФИКАЦИИ ---")
print("Для нового объекта с признаками x = [x1, x2, x3, x4]:")
print("1. Вычислить проекцию: y = w ⋅ x")
print(f"   где w = [{', '.join(f'{wi:.4f}' for wi in w)}]")
print("2. Сравнить y с порогом T:")
print(f"   - Если y ≥ {threshold:.4f}, то объект относится к классу '{class1}'")
print(f"   - Если y < {threshold:.4f}, то объект относится к классу '{class2}'")