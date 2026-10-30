import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

# Загрузка датасета (в новых версиях sklearn load_boston удален)
try:
    # Пытаемся скачать классический датасет Бостона (13 признаков)
    data = datasets.fetch_openml(name='boston', version=1, as_frame=False)
    X, y = data.data, data.target
    print("Загружен датасет Boston (13 признаков).")
except:
    # Если нет интернета, используем встроенный Калифорнийский (8 признаков)
    data = datasets.fetch_california_housing()
    X, y = data.data, data.target
    print("Загружен датасет California (8 признаков).")

# Перемешивание данных
X, y = shuffle(X, y, random_state=7)

# Разбивка данных на обучающий и тестовый наборы (80/20)
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Тестовая точка (первые 13 значений, если датасет Бостона)
# Для Калифорнийского датасета размерность будет 8
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]

# Список параметров для тестирования (для отчета)
params_list = [
    {'C': 1.0, 'epsilon': 0.1},
    {'C': 100.0, 'epsilon': 0.1},
    {'C': 1.0, 'epsilon': 1.0}
]

print("\n###### Performance for different C and epsilon ####")
for params in params_list:
    C_val = params['C']
    eps_val = params['epsilon']
    
    print(f"\n--- C={C_val}, epsilon={eps_val} ---")
    
    # Создание и обучение модели SVM
    sv_regressor = SVR(kernel='linear', C=C_val, epsilon=eps_val)
    sv_regressor.fit(X_train, y_train)
    
    # Оценка эффективности
    y_test_pred = sv_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)
    
    print(f"Mean squared error = {mse:.2f}")
    print(f"Explained variance score = {evs:.2f}")
    
    # Предсказание для конкретной точки
    # (Обрезаем test_data, если загрузился датасет на 8 признаков)
    if len(X[0]) == 8:
        pred_price = sv_regressor.predict([test_data[:8]])[0]
    else:
        pred_price = sv_regressor.predict([test_data])[0]
        
    print(f"Predicted price: {pred_price:.2f}")