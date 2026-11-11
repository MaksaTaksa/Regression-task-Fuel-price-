# Цены на топливо, регрессия

## Датасет

github не готов принять файл больше 100 мб

Ссылка на датасет: https://www.kaggle.com/datasets/alessandrolobello/gasoline-hourly-price-tracker-from-2022?select=Fuel_Station_Information.csv

## Постановка задачи
Нужно предсказать цену топлива на заправках Италии, таргет Price. Данные за 2022 год, история изменения цен и информация о заправках, после объединения 2.45 млн записей

Признаки:
- Id - номер заправки
- isSelf - колонка самообслуживания или с оператором
- Date - дата и время изменения цены
- Fuel_station_manager - управляющая компания
- Petrol_company - бренд
- Type - обычная заправка или на автомагистрали
- Station_name - название станции
- City - город
- Latitude, Longitudine - координаты

## EDA
В данных почти 60 тысяч дубликатов, их убрал. Station_name убрал, там почти 18 тысяч уникальных значений, это лишний шум. Редкие города и управляющие компании объединил в Other

Price распределена нормально, цены обновляются в основном утром и в полночь, а по году есть два пика изменений. У колонок с оператором есть выбросы около 4.0, похоже на заглушку. На удаленных заправках цены выше

Сильнее всего с ценой связаны isSelf, Date, Fuel_station_manager и City. Fuel_station_manager заметно улучшает MAE, City почти ничего не дает, координаты полезнее города

Дату перевел в день года, время в минуты от начала суток. City закодировал частотой, Fuel_station_manager и Petrol_company через TargetEncoder, Type через OneHot

## Baseline
Сравнил DummyRegressor и LightGBM без настройки:
- Dummy: R2 0, MAE 0.146
- LightGBM: R2 0.837, MAE 0.046

Тут данные разбивались случайно, поэтому оценка завышена, дальше разбивал по времени

## Гиперпараметры
Пробовал Lasso, Ridge, ElasticNet и XGBRFRegressor, но они были хуже, поэтому оставил LightGBM, XGBoost и CatBoost. Подбирал через Optuna, по 30 триалов на модель, первые 80% данных по времени на обучение, последние 20% на тест

Лучшие параметры:
- LightGBM: learning_rate 0.29, n_estimators 800, min_child_samples 300, num_leaves 161
- XGBoost: learning_rate 0.045, n_estimators 1000, max_depth 7, reg_lambda 0.014
- CatBoost: learning_rate 0.29, depth 6, l2_leaf_reg 10, iterations 900

Результаты на тесте:
- LightGBM: R2 0.556
- XGBoost: R2 0.522
- CatBoost: R2 0.504

Лучше всех LightGBM

## Improvements
Добавил price_lag, это предыдущая цена этой же заправки. Цены меняются редко и небольшими шагами, так что прошлая цена хорошо описывает текущую

Еще добавил регион, сделал KMeans на 10 кластеров по координатам заправок. Хотел через dbscan, но решил, что будет чересчур

С этими признаками LightGBM дал R2 0.788, то есть R2 вырос на 0.23

Пробовал ускорить обучение через PySpark, но это ничего не дало, помогло убрать KNNImputer и лямбда функции

## Файлы
- exploration_data_analysis.ipynb - EDA
- base_line.ipynb - baseline
- hptunning.ipynb - подбор гиперпараметров
- improvments.ipynb - новые признаки и итоговая модель
- funcs.py - функции предобработки
- final_result.joblib - итоговая модель
