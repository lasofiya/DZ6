import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import json

# Загрузка данных
# Убедитесь, что путь к файлу правильный
data = pd.read_csv('sdss_redshift.csv')

# Выбор признаков и целевой переменной
X = data[['u', 'g', 'r', 'i', 'z']].values
y = data['redshift'].values

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Оптимизация гиперпараметров с помощью GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

# GridSearchCV для поиска лучших гиперпараметров
grid_search = GridSearchCV(RandomForestRegressor(
), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Получение лучших гиперпараметров
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Создание и обучение случайного леса с оптимальными параметрами
forest = RandomForestRegressor(**best_params)
forest.fit(X_train, y_train)

# Предсказания на обучающей и тестовой выборках
train_predictions = forest.predict(X_train)
test_predictions = forest.predict(X_test)

# Вычисление стандартного отклонения
train_std = np.std(y_train - train_predictions)
test_std = np.std(y_test - test_predictions)

# Сохранение результатов в JSON файл
results = {"train": train_std, "test": test_std}
with open('redshift.json', 'w') as f:
    json.dump(results, f)

# Построение графика "истинное значение — предсказание"
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.savefig('redshift.png')

# Предсказания для данных без красных смещений
unknown_data = pd.read_csv('sdss.csv')
unknown_X = unknown_data[['u', 'g', 'r', 'i', 'z']].values
unknown_predictions = forest.predict(unknown_X)

# Сохранение предсказаний в CSV файл под другим именем
output_file = 'sdss_predict_new.csv'
unknown_data['redshift'] = unknown_predictions
unknown_data.to_csv(output_file, index=False)
