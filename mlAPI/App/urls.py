from django.contrib import admin
from django.urls import path, include
from . import views

app_name = 'App'

urlpatterns = [
    path('train', views.Train.as_view(), name='train'),
    path('predict', views.Predict.as_view(), name='predict'),
    path('', views.home, name='home'),
]
