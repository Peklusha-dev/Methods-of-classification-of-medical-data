# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# %%
g1 = np.array(
	[
		[0.7, 0.3, 1.2],
		[0.5, 0.7, 1.0],
		[0.4, 1.0, 0.4],
		[0.7, 0.7, 1.0],
		[0.6, 0.6, 1.5],
		[0.6, 0.6, 1.2],
		[0.6, 0.5, 1.0],
		[0.4, 0.9, 0.6],
		[0.5, 0.6, 1.1],
		[0.8, 0.3, 1.2],
	]
)

g2 = np.array(
	[
		[0.4, 0.2, 0.8],
		[0.2, 0.2, 0.7],
		[0.9, 0.3, 0.5],
		[0.8, 0.3, 0.6],
		[0.5, 0.6, 0.4],
		[0.6, 0.5, 0.7],
		[0.4, 0.4, 1.2],
		[0.6, 0.3, 1.0],
		[0.3, 0.2, 0.6],
		[0.5, 0.5, 0.8],
	]
)

# %% [markdown]
# # Задание 1: Метод главных компонент

# %%
# Объединение данных
X = np.vstack([g1, g2])
g = np.array([0] * len(g1) + [1] * len(g2))

# Применение PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Объясненная дисперсия главных компонент:", pca.explained_variance_ratio_)
print("Суммарная объясненная дисперсия:", sum(pca.explained_variance_ratio_))

# %%
# Диаграмма рассеяния
plt.figure(figsize=(10, 8))
for label, marker, color in zip([0, 1], ["o", "^"], ["blue", "red"]):
	idx = g == label
	plt.scatter(
		X_pca[idx, 0],
		X_pca[idx, 1],
		marker=marker,
		color=color,
		label=f"Класс {label + 1}",
		alpha=0.7,
		s=100
	)

plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("Распределение классов в пространстве двух главных компонент")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Вывод о линейной разделимости:
# По диаграмме рассеяния в пространстве главных компонент можно видеть, что классы частично перекрываются,
# но в целом наблюдается тенденция к разделимости. Класс 1 (синий) в основном расположен в одной области,
# а класс 2 (красный) - в другой, однако есть некоторое перекрытие в центральной области.

# %% [markdown]
# # Задание 2: Процедура последовательного обучения

# %%
c = 1
A = g1.copy()
B = g2.copy()

n = A.shape
nRows = n[0]
nColomns = n[1]

# Добавляем столбец единиц для смещения
A = np.hstack([A, np.ones((nRows, 1))])
B = np.hstack([B, np.ones((nRows, 1))])

# Инвертируем класс B
BB = B * (-1)
y = np.vstack([A, BB])

# Начальное приближение - нормализованная разность средних
w = np.mean(A, axis=0) - np.mean(B, axis=0)
w = w / np.linalg.norm(w)

print("Начальный весовой вектор:", w)

# %%
# Алгоритм последовательного обучения с коррекцией ошибок
cont = True
iterations = 0
max_iterations = 1000  # защита от бесконечного цикла

while cont and iterations < max_iterations:
	iterations += 1
	errors = 0

	for i in range(2 * nRows):
		proekc_i = np.dot(y[i, :], w)
		if proekc_i < 0:
			w = w + c * y[i, :]
			errors += 1

	proekc = np.dot(y, w)

	if np.min(proekc) >= 0:
		cont = False
		print(f"Обучение завершено за {iterations} итераций")
	elif iterations % 100 == 0:
		print(f"Итерация {iterations}, ошибок: {errors}")

if iterations == max_iterations:
	print("Достигнуто максимальное количество итераций")

print("Финальный весовой вектор:", w)

# %% [markdown]
# # Задание 3: Проекция классов и анализ ошибок

# %%
# Проекция классов на весовой вектор
arr1 = A.copy()
arr2 = B.copy()

x1 = arr1 @ w
x2 = arr2 @ w

# Построение гистограмм
plt.figure(figsize=(12, 6))

t = np.arange(-3, 3, 0.1)
counts1, bins1 = np.histogram(x1, bins=t)
counts2, bins2 = np.histogram(x2, bins=t)

plt.plot((bins1[1:] + bins1[:-1]) / 2, counts1, 'b-', linewidth=2, label="Класс 1")
plt.plot((bins2[1:] + bins2[:-1]) / 2, counts2, 'r-', linewidth=2, label="Класс 2")
plt.axvline(x=0, color='k', linestyle='--', label='Граница решения')
plt.xlabel('Проекция на весовой вектор')
plt.ylabel('Частота')
plt.title('Гистограммы проекций классов на весовой вектор')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# %%
# Алгоритм распознавания
def classify_sample(sample, w):
	"""Классификация нового образца"""
	extended_sample = np.append(sample, 1)  # добавляем смещение
	projection = np.dot(extended_sample, w)
	return 0 if projection >= 0 else 1  # класс 1 или 2


# %%
# Оценка ошибок классификации
def calculate_error_rate(A_orig, B_orig, w):
	"""Расчет ошибки классификации"""
	errors = 0
	total_samples = len(A_orig) + len(B_orig)

	# Проверка класса 1
	for sample in A_orig:
		if classify_sample(sample, w) != 0:
			errors += 1
			print(f"Ошибка: образец {sample} класса 1 классифицирован как класс 2")

	# Проверка класса 2
	for sample in B_orig:
		if classify_sample(sample, w) != 1:
			errors += 1
			print(f"Ошибка: образец {sample} класса 2 классифицирован как класс 1")

	error_rate = errors / total_samples
	print(f"\nРезультаты классификации:")
	print(f"Ошибок классификации: {errors}/{total_samples}")
	print(f"Процент ошибок: {error_rate:.2%}")
	return error_rate


# %%
# Расчет ошибок
error_rate = calculate_error_rate(g1, g2, w)

# %%
# Визуализация разделяющей гиперплоскости в пространстве первых двух главных компонент
plt.figure(figsize=(10, 8))

# Данные в пространстве PCA
for label, marker, color in zip([0, 1], ["o", "^"], ["blue", "red"]):
	idx = g == label
	plt.scatter(
		X_pca[idx, 0],
		X_pca[idx, 1],
		marker=marker,
		color=color,
		label=f"Класс {label + 1}",
		alpha=0.7,
		s=100
	)

# Построение разделяющей линии в пространстве PCA
xx = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)

# Для визуализации разделяющей линии в пространстве PCA используем другой подход
# Создаем сетку точек в пространстве PCA
x_min, x_max = X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1
y_min, y_max = X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Преобразуем точки обратно в исходное пространство
mesh_points = np.c_[xx.ravel(), yy.ravel()]
mesh_points_original = pca.inverse_transform(mesh_points)

# Классифицируем каждую точку
Z = np.array([classify_sample(point, w) for point in mesh_points_original])
Z = Z.reshape(xx.shape)

# Рисуем контур разделяющей границы
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)

plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("Разделяющая граница в пространстве главных компонент")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Тестирование на новых данных (пример)
print("\nТестирование на новых данных:")
test_samples = [
	[0.6, 0.5, 1.1],  # похож на класс 1
	[0.3, 0.3, 0.7],  # похож на класс 2
	[0.5, 0.5, 0.9],  # пограничный случай
]

for i, sample in enumerate(test_samples):
	classification = classify_sample(sample, w)
	projection = np.dot(np.append(sample, 1), w)
	print(f"Образец {i + 1}: {sample} -> Класс {classification + 1} (проекция: {projection:.3f})")

# %%
# Анализ разделимости по проекциям
print("\nАнализ разделимости:")
print(f"Минимальная проекция класса 1: {x1.min():.3f}")
print(f"Максимальная проекция класса 2: {x2.max():.3f}")
print(f"Перекрытие проекций: {'Да' if x1.min() < x2.max() else 'Нет'}")

# %%
# Сводка результатов
print("\n" + "=" * 50)
print("СВОДКА РЕЗУЛЬТАТОВ")
print("=" * 50)
print(f"1. Объясненная дисперсия PCA: {sum(pca.explained_variance_ratio_):.2%}")
print(f"2. Количество итераций обучения: {iterations}")
print(f"3. Точность классификации: {(1 - error_rate):.2%}")
print(f"4. Весовой вектор: {w}")
print("=" * 50)