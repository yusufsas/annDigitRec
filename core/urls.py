





from django.contrib import admin
from django.urls import path
from django.urls import path,include
from .views import index
from django.conf.urls.static import static
from django.conf import settings
urlpatterns = [
    path('',index,name='index'),
]