from django.conf import settings
from django.urls import include, path, re_path
from django.views.static import serve

urlpatterns = [
    path("", include("console.urls")),
    # a local tool: serve its own CSS and JS, whatever flags runserver was started with
    re_path(r"^static/(?P<path>.*)$", serve, {"document_root": settings.BASE_DIR / "console" / "static"}),
]
