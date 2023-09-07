from django.urls import path

from . import views
#控制页面url路径
app_name = 'myapp'
urlpatterns = [

    path('', views.answer, name='answer'),
    path('', views.answer, name='index'),

]