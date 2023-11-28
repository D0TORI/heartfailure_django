from django.urls import path
from django.conf.urls import include
from . import views

urlpatterns = [
    path('', views.mainpage, name='mainpage'),
    path('credit/', views.credit, name='credit'),
    path('submit/', views.submit_data, name='submit_data'),
]