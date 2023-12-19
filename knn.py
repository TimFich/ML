import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# датасет
iris = datasets.load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names

# нормализация данных (встроенный метод, который центрирует и масштабирует данные)
scaler = StandardScaler()

data_normalized = scaler.fit_transform(data)

# Разделение на обучающую и тестовую выборку
# случайным образом разделяет данные
X_train, X_test, y_train, y_test = train_test_split(data_normalized, target, test_size=0.3)

# поиск оптимального k через метод score через тестовую выборку
scores = []
for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

optimal_k = np.argmax(scores) + 1
print(f"Оптимальное значение k: {optimal_k}")


# вывод 2.0
def plot_projections(X, y, title):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    combinations = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for (i, j), ax in zip(combinations, axs.ravel()):
        ax.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.jet)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


plot_projections(data, target, "Проекции до нормализации")
plot_projections(data_normalized, target, "Проекции после нормализации")

# input
new_sample = np.array([[float(input(f"Введите значение для {name}: ")) for name in feature_names]])
new_sample_normalized = scaler.transform(new_sample)

knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(data_normalized, target)
prediction = knn.predict(new_sample_normalized)

print(f"Класс нового объекта: {iris.target_names[prediction[0]]}")

