from django.urls import path

from . import views

urlpatterns = [
    path("", views.forecast, name="forecast"),
    path("operations", views.operations, name="operations"),
    path("operations/run/<str:key>", views.run, name="run"),
    path("operations/job/<str:job_id>", views.job, name="job"),
    path("activity", views.activity, name="activity"),
    path("performance", views.performance, name="performance"),
    path("model", views.model, name="model"),
]
