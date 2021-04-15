from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from django.shortcuts import render
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from catboost import CatBoostRegressor


# Отрисовка главной страницы
def home(request):
    return render(request, 'file_input.html')


# Тренировка модели на полученных данных
class Train(views.APIView):
    def post(self, request):
        try:
            df = pd.read_csv(request.FILES['file'])
            catboost = CatBoostRegressor(loss_function='MAPE', depth=9, iterations=2500, l2_leaf_reg=8,
                                         random_state=42, silent=True)
            normalizer = Normalizer()

            df['date'] = pd.to_datetime(df['date'])
            x = df.drop('y', axis=1)
            y = df['y']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42, shuffle=False)
            x_train_without_date = x_train.drop('date', axis=1)

            cols = x_train_without_date.columns
            normalized_values = normalizer.fit_transform(x_train_without_date)
            x_train_without_date = pd.DataFrame(normalized_values, columns=cols)
            x_train_norm = x_train_without_date
            x_train_norm['date'] = x_train['date']

            catboost.fit(x_train_norm, y_train)

            result = catboost.score(x_test, y_test)

            # Сохранение модели
            filename = 'model.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(catboost, file)

        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        return Response(result, status=status.HTTP_200_OK)


# Предсказание на полученных данных
class Predict(views.APIView):
    def post(self, request):
        try:
            # Загрузка раннее сохраненной модели
            filename = 'model.pkl'
            with open(filename, 'rb') as file:
                catboost = pickle.load(file)

            df = pd.read_csv(request.FILES['file'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop('y', axis=1)

            result = catboost.predict(df)

        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        return Response(result, status=status.HTTP_200_OK)
