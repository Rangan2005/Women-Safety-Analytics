
from django.urls import path
from .views import display_live_feed,live_feed

urlpatterns = [
    path('live/', display_live_feed, name='display_live_feed'),
    path('live_feed/', live_feed, name='live_feed'),
]
