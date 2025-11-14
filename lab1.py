import numpy as np
from sklearn.datasets import load_iris
import pandas as pd


def main():
	# Генерация случайных массивов
	print("Генерация массивов:")
	arr1 = np.random.normal(6, 1, 10)
	arr2 = np.random.normal(7, 1, 10)

	print("arr1:", arr1)
	print("arr2:", arr2)
	print()

	# Статистика массивов
	print("Статистика arr1:")
	print("Среднее:", arr1.mean(), "Стандартное отклонение:", arr1.std())
	print("Статистика arr2:")
	print("Среднее:", arr2.mean(), "Стандартное отклонение:", arr2.std())
	print()

	# Корреляция
	corr = np.corrcoef(arr1, arr2)
	print("Матрица корреляции:")
	print(corr)
	print()

	# Работа с данными Iris
	print("Загрузка набора данных Iris:")
	ds = load_iris()
	data = ds.data
	feat_names = ds.feature_names
	target = ds.target

	# Создание DataFrame
	full_df = pd.DataFrame(
		np.hstack([data, np.expand_dims(target, axis=1)]),
		columns=list(feat_names) + ["target"]
	)

	# Фильтрация setosa
	setosa_df = full_df[full_df["target"] == 0]

	print("Первые 5 строк setosa (длина и ширина чашелистика):")
	print(setosa_df[feat_names[:2]].head(5))
	print()

	print("Описательная статистика setosa:")
	print(setosa_df[feat_names[:2]].describe())

	print("Корреляция между первыми двумя признаками Setosa:")
	corr_setosa = np.corrcoef(setosa_df[feat_names[0]], setosa_df[feat_names[1]])
	print(corr_setosa)
	print(f"Коэффициент корреляции: {corr_setosa[0, 1]:.3f}")


if __name__ == "__main__":
	main()