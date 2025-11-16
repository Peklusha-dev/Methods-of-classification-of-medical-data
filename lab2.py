import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from scipy.stats import gaussian_kde  # Добавь эту строку

# 1. Загрузка данных ирисов Фишера
ds = load_iris()
data = ds.data
target = ds.target

# Берем только setosa и versicolor
data = data[:100]
target = target[:100]  # ОБЯЗАТЕЛЬНО обрезать target тоже!

# 2. Построение диаграмм hist и plot по 3-му признаку (длина лепестка)
plt.figure(figsize=(15, 5))

# Histogram
plt.subplot(1, 2, 1)
setosa_petal_length = data[target == 0, 2]
versicolor_petal_length = data[target == 1, 2]

plt.hist(setosa_petal_length, bins=15, alpha=0.6, label='setosa', color='blue')
plt.hist(versicolor_petal_length, bins=15, alpha=0.6, label='versicolor', color='red')
plt.xlabel('Длина лепестка (cm)')
plt.ylabel('Частота')
plt.title('Hist: Длина лепестка для setosa и versicolor')
plt.legend()
plt.grid(True)

# Plot
plt.subplot(1, 2, 2)

for cls, color in zip(["setosa", "versicolor"], ["blue", "red"]):
    if cls == "setosa":
        subset = setosa_petal_length
    else:
        subset = versicolor_petal_length

    # Сортируем значения и создаем гистограмму
    sorted_subset = np.sort(subset)
    count_arr, val_arr = np.histogram(sorted_subset, bins=15)

    # Строим график по точкам гистограммы
    plt.plot(val_arr[1:], count_arr, label=cls, color=color, linewidth=2)

plt.xlabel('Длина лепестка (cm)')
plt.ylabel('Количество')
plt.title('Plot: Длина лепестка для setosa и versicolor')
plt.legend()
plt.grid(True)

# 3. Boxplot по 2-му признаку (ширина чашелистика)
plt.figure(figsize=(8, 6))
setosa_sepal_width = data[target == 0, 1]
versicolor_sepal_width = data[target == 1, 1]

plt.boxplot([setosa_sepal_width, versicolor_sepal_width],
            labels=['setosa', 'versicolor'])
plt.ylabel('Ширина чашелистика (cm)')
plt.title('Boxplot: Ширина чашелистика для setosa и versicolor')
plt.grid(True)
plt.show()

# 4. Создание 2 массивов
np.random.seed(42)  # Для воспроизводимости

# Параметры варианта:
# Массив 1: m = 6, σ = 1
# Массив 2: m = 7, σ = 1

# Генерация массивов по 10 чисел каждый
array1 = np.random.normal(6, 1, 30)
array2 = np.random.normal(7, 1, 30)

print("Ваш вариант - Параметры распределений:")
print(f"Массив 1: m = 6, σ = 1")
print(f"Сгенерированные значения: {array1}")
print(f"\nМассив 2: m = 7, σ = 1")
print(f"Сгенерированные значения: {array2}")

# 5. Boxplot с зазубринами для массивов
plt.figure(figsize=(12, 6))

# Boxplot для массивов
plt.subplot(1, 2, 1)
box_plot = plt.boxplot([array1, array2],
            labels=['Массив 1\n(m=6, σ=1)', 'Массив 2\n(m=7, σ=1)'],
            notch=True,
            patch_artist=True)

# Цвета для наглядности
colors = ['lightblue', 'lightcoral']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('Значения')
plt.title('Boxplot с зазубринами')
plt.grid(True)

# Гистограмма для сравнения распределений
plt.subplot(1, 2, 2)
plt.hist(array1, bins=8, alpha=0.6, label='Массив 1 (m=6, σ=1)', color='blue')
plt.hist(array2, bins=8, alpha=0.6, label='Массив 2 (m=7, σ=1)', color='red')
plt.xlabel('Значения')
plt.ylabel('Частота')
plt.title('Гистограмма распределений')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Анализ различий между массивами
print("\n" + "="*50)
print("АНАЛИЗ РАЗЛИЧИЙ МЕЖДУ МАССИВАМИ")
print("="*50)

print(f"\nСтатистика по массиву 1 (m=6, σ=1):")
print(f"Среднее значение: {np.mean(array1):.2f}")
print(f"Медиана: {np.median(array1):.2f}")
print(f"Стандартное отклонение: {np.std(array1):.2f}")
print(f"Минимальное значение: {np.min(array1):.2f}")
print(f"Максимальное значение: {np.max(array1):.2f}")
print(f"Размах: {np.ptp(array1):.2f}")

print(f"\nСтатистика по массиву 2 (m=7, σ=1):")
print(f"Среднее значение: {np.mean(array2):.2f}")
print(f"Медиана: {np.median(array2):.2f}")
print(f"Стандартное отклонение: {np.std(array2):.2f}")
print(f"Минимальное значение: {np.min(array2):.2f}")
print(f"Максимальное значение: {np.max(array2):.2f}")
print(f"Размах: {np.ptp(array2):.2f}")

print(f"\nСравнительный анализ:")
print(f"Разница средних: {np.mean(array2) - np.mean(array1):.2f}")
print(f"Разница медиан: {np.median(array2) - np.median(array1):.2f}")
print(f"Относительная разница средних: {(np.mean(array2) - np.mean(array1)) / np.mean(array1) * 100:.1f}%")

# Проверка статистической значимости различий
from scipy import stats

t_stat, p_value = stats.ttest_ind(array1, array2)
print(f"\nСтатистический тест (t-тест):")
print(f"t-статистика: {t_stat:.3f}")
print(f"p-значение: {p_value:.3f}")

if p_value < 0.05:
    print("Вывод: Различия между массивами статистически значимы (p < 0.05)")
else:
    print("Вывод: Различия между массивами не статистически значимы (p ≥ 0.05)")

# Анализ перекрытия распределений
overlap_min = max(np.min(array1), np.min(array2))
overlap_max = min(np.max(array1), np.max(array2))
overlap_range = max(0, overlap_max - overlap_min)

print(f"\nАнализ перекрытия распределений:")
print(f"Диапазон перекрытия: {overlap_range:.2f}")
print(f"Процент перекрытия от общего диапазона: {overlap_range / (max(np.max(array1), np.max(array2)) - min(np.min(array1), np.min(array2))) * 100:.1f}%")