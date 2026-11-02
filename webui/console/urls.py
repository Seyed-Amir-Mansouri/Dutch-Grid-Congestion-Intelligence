from django.urls import path

from . import views

urlpatterns = [
    path("", views.forecast, name="forecast"),
    path("operations", views.operations, name="operations"),
    path("operations/run/<str:key>", views.run, name="run"),
    path("operations/job/<str:job_id>", views.job, name="job"),
    path("operations/job/<str:job_id>/log", views.job_log, name="job_log"),
    path("operations/job/<str:job_id>/stop", views.stop, name="stop"),
    path("operations/clear/<str:kind>", views.clear, name="clear"),
    path("operations/drive", views.drive, name="drive"),
    path("activity", views.activity, name="activity"),
    path("performance", views.performance, name="performance"),
    path("model", views.model, name="model"),
]
