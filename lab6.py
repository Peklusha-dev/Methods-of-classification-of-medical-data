# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import load_iris
import pandas as pd
from collections import Counter

# %% [markdown]
# # Задание 1: Линейная классификация массива a на классы A, B, C

# %%
# Генерация данных
np.random.seed(42)
a11 = np.random.normal(5, 1, 10)
a12 = np.random.normal(4, 1, 10)
a13 = np.random.normal(7, 1, 10)
a14 = np.random.normal(2, 1, 10)
a1 = np.column_stack([a11, a12, a13, a14])

a21 = np.random.normal(3, 2, 10)
a22 = np.random.normal(5, 2, 10)
a23 = np.random.normal(4, 2, 10)
a24 = np.random.normal(3, 2, 10)
a2 = np.column_stack([a21, a22, a23, a24])

a31 = np.random.normal(7, 1, 10)
a32 = np.random.normal(6, 1, 10)
a33 = np.random.normal(3, 1, 10)
a34 = np.random.normal(2, 1, 10)
a3 = np.column_stack([a31, a32, a33, a34])

a = np.vstack([a1, a2, a3])
gg = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
g = np.array(gg)

print("Размерность массива a:", a.shape)
print("Классы:", Counter(g))

# %%
# Линейный дискриминантный анализ
lda = LinearDiscriminantAnalysis(
    solver="svd",
    shrinkage=None,
    store_covariance=False,
    tol=1.0e-4,
)
lda.fit(a, g)
g_pred_lda = lda.predict(a)

print("LDA классификация:")
print("Предсказанные классы:", g_pred_lda)
errors_lda = g_pred_lda != g
print("Ошибки классификации:", Counter(g[errors_lda]))
print("Общее количество ошибок:", np.sum(errors_lda))
print("Точность LDA: {:.1f}%".format((1 - np.mean(errors_lda)) * 100))

# %%
# Визуализация LDA (первые два признака)
plt.figure(figsize=(10, 6))
for label, marker, color in zip(["A", "B", "C"], ["o", "s", "^"], ["red", "green", "blue"]):
    idx = g == label
    plt.scatter(a[idx, 0], a[idx, 1], marker=marker, color=color, label=f"True {label}", alpha=0.7, s=80)

# Отметка ошибок
error_count = 0
for i, (true, pred) in enumerate(zip(g, g_pred_lda)):
    if true != pred:
        plt.scatter(a[i, 0], a[i, 1], color="black", marker="x", s=100, linewidth=2,
                   label="Ошибка" if error_count == 0 else "")
        error_count += 1

plt.title("LDA классификация массива a (первые два признака)")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Задание 2: Квадратичная классификация массива a на классы A, B, C

# %%
# Квадратичный дискриминантный анализ
qda = QuadraticDiscriminantAnalysis()
qda.fit(a, g)
g_pred_qda = qda.predict(a)

print("QDA классификация:")
print("Предсказанные классы:", g_pred_qda)
errors_qda = g_pred_qda != g
print("Ошибки классификации:", Counter(g[errors_qda]))
print("Общее количество ошибок:", np.sum(errors_qda))
print("Точность QDA: {:.1f}%".format((1 - np.mean(errors_qda)) * 100))

# %%
# Визуализация QDA (первые два признака)
plt.figure(figsize=(10, 6))
for label, marker, color in zip(["A", "B", "C"], ["o", "s", "^"], ["red", "green", "blue"]):
    idx = g == label
    plt.scatter(a[idx, 0], a[idx, 1], marker=marker, color=color, label=f"True {label}", alpha=0.7, s=80)

# Отметка ошибок
error_count = 0
for i, (true, pred) in enumerate(zip(g, g_pred_qda)):
    if true != pred:
        plt.scatter(a[i, 0], a[i, 1], color="black", marker="x", s=100, linewidth=2,
                   label="Ошибка" if error_count == 0 else "")
        error_count += 1

plt.title("QDA классификация массива a (первые два признака)")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Задание 3: Линейная классификация ирисов Фишера

# %%
# Загрузка данных ирисов Фишера
ds = load_iris()
data = ds.data
feat_names = ds.feature_names
target = ds.target
target_names = ds.target_names

print("Информация о наборе данных Iris:")
print("Размерность данных:", data.shape)
print("Признаки:", feat_names)
print("Классы:", target_names)
print("Количество образцов по классам:", Counter(target))

# %%
# Линейная классификация ирисов
iris_lda = LinearDiscriminantAnalysis(
    solver="svd",
    shrinkage=None,
    store_covariance=False,
    tol=1.0e-4,
)
iris_lda.fit(data, target)
pred_iris_lda = iris_lda.predict(data)

errors_iris_lda = pred_iris_lda != target
print("LDA классификация ирисов:")
print("Ошибки по классам:", Counter(target[errors_iris_lda]))
print("Общее количество ошибок:", np.sum(errors_iris_lda))
print("Точность LDA: {:.1f}%".format((1 - np.mean(errors_iris_lda)) * 100))

# %%
# Визуализация LDA для ирисов (первые два признака)
plt.figure(figsize=(10, 6))
for label, marker, color in zip([0, 1, 2], ["o", "s", "^"], ["red", "green", "blue"]):
    idx = target == label
    plt.scatter(data[idx, 0], data[idx, 1], marker=marker, color=color,
               label=f"True {target_names[label]}", alpha=0.7, s=60)

# Отметка ошибок
error_count = 0
for i, (true, pred) in enumerate(zip(target, pred_iris_lda)):
    if true != pred:
        plt.scatter(data[i, 0], data[i, 1], color="black", marker="x", s=80, linewidth=2,
                   label="Ошибка" if error_count == 0 else "")
        error_count += 1

plt.title("LDA классификация ирисов Фишера (sepal length vs sepal width)")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Задание 4: Квадратичная классификация ирисов Фишера

# %%
# Квадратичная классификация ирисов
iris_qda = QuadraticDiscriminantAnalysis()
iris_qda.fit(data, target)
pred_iris_qda = iris_qda.predict(data)

errors_iris_qda = pred_iris_qda != target
print("QDA классификация ирисов:")
print("Ошибки по классам:", Counter(target[errors_iris_qda]))
print("Общее количество ошибок:", np.sum(errors_iris_qda))
print("Точность QDA: {:.1f}%".format((1 - np.mean(errors_iris_qda)) * 100))

# %%
# Визуализация QDA для ирисов (первые два признака)
plt.figure(figsize=(10, 6))
for label, marker, color in zip([0, 1, 2], ["o", "s", "^"], ["red", "green", "blue"]):
    idx = target == label
    plt.scatter(data[idx, 0], data[idx, 1], marker=marker, color=color,
               label=f"True {target_names[label]}", alpha=0.7, s=60)

# Отметка ошибок
error_count = 0
for i, (true, pred) in enumerate(zip(target, pred_iris_qda)):
    if true != pred:
        plt.scatter(data[i, 0], data[i, 1], color="black", marker="x", s=80, linewidth=2,
                   label="Ошибка" if error_count == 0 else "")
        error_count += 1

plt.title("QDA классификация ирисов Фишера (sepal length vs sepal width)")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Сравнение результатов

# %%
# Сводная таблица результатов
print("=" * 50)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 50)

results = {
    "Массив a - LDA": {
        "Ошибки": np.sum(errors_lda),
        "Точность": (1 - np.mean(errors_lda)) * 100
    },
    "Массив a - QDA": {
        "Ошибки": np.sum(errors_qda),
        "Точность": (1 - np.mean(errors_qda)) * 100
    },
    "Ирисы - LDA": {
        "Ошибки": np.sum(errors_iris_lda),
        "Точность": (1 - np.mean(errors_iris_lda)) * 100
    },
    "Ирисы - QDA": {
        "Ошибки": np.sum(errors_iris_qda),
        "Точность": (1 - np.mean(errors_iris_qda)) * 100
    }
}

for method, result in results.items():
    print(f"{method:20} | Ошибки: {result['Ошибки']:2d} | Точность: {result['Точность']:5.1f}%")