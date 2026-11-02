from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from . import charts, data, jobs

ASSETS = Path(__file__).parent / "static" / "console"


def _common(request, page: str) -> dict:
    """Values every page shows in its header."""
    bundle = data.bundle()
    return {
        "page": page,
        "model_version": bundle["model_version"] if bundle else None,
        "model_name": ("Predictor + Inspector" if bundle.get("inspector") else "Predictor") if bundle else None,
        "model_built": pd.Timestamp(bundle["trained_at"]).tz_convert("Europe/Amsterdam") if bundle else None,
        "threshold": bundle["day_threshold"] if bundle else None,
        "running": jobs.busy(),
        # changes whenever a stylesheet or script changes, so browsers never keep an old copy
        "assets": int(max(f.stat().st_mtime for f in ASSETS.glob("*.*"))),
    }


def _number(value) -> float | None:
    """A float, or None for missing values."""
    return None if value is None or pd.isna(value) else float(value)


def forecast(request):
    """The forecast for one delivery day: map, counts and the flagged areas."""
    context = _common(request, "forecast")
    days = data.forecast_days()
    context["days"] = days
    if not days:
        return render(request, "console/forecast.html", context)
    chosen = request.GET.get("day")
    day = pd.Timestamp(chosen) if chosen else days[0]
    if day not in days:
        raise Http404("No forecast for that day")
    rows = data.forecast(day)
    records = [
        {**r, "peak_probability": float(r["peak_probability"]), "inspector_confidence": _number(r["inspector_confidence"])}
        for r in rows.to_dict("records")
    ]
    settled = rows["had_request"].notna().all() and len(rows)
    for i, r in enumerate(records):
        r["id"] = i
    flagged = [r for r in records if r["status"] != "clear"]
    for r in flagged:
        r["peak_meter"] = charts.meter(r["peak_probability"])
        r["inspector_meter"] = charts.meter(r["inspector_confidence"])
        r["icon"] = charts.legend_icon(r["status"])
    counts = rows["status"].value_counts()
    context.update(
        day=day,
        index=days.index(day),
        previous=days[days.index(day) + 1] if days.index(day) + 1 < len(days) else None,
        next=days[days.index(day) - 1] if days.index(day) > 0 else None,
        issued=pd.Timestamp(rows["issued_at"].max()).tz_convert("Europe/Amsterdam") if len(rows) else None,
        active=len(rows),
        alerts=int(counts.get("alert", 0)),
        vetoed=int(counts.get("vetoed", 0)),
        flagged=flagged,
        settled=bool(settled),
        outcome_correct=int(((rows["status"] == "alert") & (rows["had_request"] == 1)).sum()) if settled else None,
        outcome_requests=int((rows["had_request"] == 1).sum()) if settled else None,
        map_points=[
            {"id": i, "area": r["area"], "lat": _number(r["lat"]), "lon": _number(r["lon"]), "status": r["status"],
             "cell": r["cell"] if isinstance(r["cell"], str) else None, "peak": r["peak_probability"],
             "inspector": r["inspector_confidence"], "window": f"{r['window_start']}–{r['window_end']}",
             "request": None if pd.isna(r["had_request"]) else int(r["had_request"])}
            for i, r in enumerate(records) if _number(r["lat"]) is not None
        ],
        legend=[(charts.legend_icon(k), v["label"]) for k, v in charts.STATUS_STYLE.items()],
    )
    return render(request, "console/forecast.html", context)


def operations(request):
    """The steps an operator can run, the running job and recent jobs."""
    context = _common(request, "operations")
    selected = request.GET.get("job")
    recent = jobs.history()
    files = data.file_status()
    listed = [f for g in files for f in g["files"]]
    context.update(
        steps=jobs.steps(),
        entsoe_key=jobs.entsoe_key_set(),
        recent=recent,
        selected=jobs.status(selected) if selected else (recent[0] if recent else None),
        files=files,
        files_found=sum(f["exists"] for f in listed),
        files_total=len(listed),
        drive_ready=jobs.drive_ready(),
        error=request.GET.get("error"),
        done=request.GET.get("done"),
    )
    return render(request, "console/operations.html", context)


@require_POST
def run(request, key: str):
    """Start a step and show its log."""
    found = {s["key"]: s for s in jobs.steps()}
    if key not in found:
        raise Http404
    if key == "entsoe" and not jobs.entsoe_key_set():
        return redirect("/operations?" + urlencode({"error": "Set ENTSOE_API_KEY before starting the web UI"}))
    if found[key]["missing"]:
        return redirect("/operations?" + urlencode({"error": "; ".join(found[key]["missing"])}))
    try:
        meta = jobs.start(key, found[key]["title"], request.POST.dict())
    except (RuntimeError, ValueError, KeyError) as error:
        return redirect("/operations?" + urlencode({"error": str(error) or "Missing form value"}))
    return redirect(f"/operations?job={meta['id']}")


@require_POST
def clear(request, kind: str):
    """Delete the data and outputs, or with kind "reset" the model as well."""
    if kind not in ("data", "reset"):
        raise Http404
    try:
        count = jobs.clear(with_model=kind == "reset")
    except RuntimeError as error:
        return redirect("/operations?" + urlencode({"error": str(error)}))
    what = "Data, outputs, job history and model" if kind == "reset" else "Data, outputs and job history"
    return redirect("/operations?" + urlencode({"done": f"{what} cleared: {count} files deleted"}))


@require_POST
def drive(request):
    """Download the shared data and model files from Google Drive and show the job."""
    if not jobs.drive_ready():
        return redirect("/operations?" + urlencode({"error": "Add the Drive links to the [drive] section of config.toml"}))
    try:
        meta = jobs.start("drive", "Download from Drive", {})
    except RuntimeError as error:
        return redirect("/operations?" + urlencode({"error": str(error)}))
    return redirect(f"/operations?job={meta['id']}")


@require_POST
def stop(request, job_id: str):
    """Stop the running job and show it."""
    try:
        jobs.stop(job_id)
    except RuntimeError as error:
        return redirect("/operations?" + urlencode({"job": job_id, "error": str(error)}))
    return redirect("/operations?" + urlencode({"job": job_id}))


def job(request, job_id: str):
    """A job's state and the end of its log, for live updates."""
    meta = jobs.status(job_id)
    if meta is None:
        raise Http404
    return JsonResponse({"status": meta["status"], "step": meta.get("step"), "steps": len(meta["commands"]),
                         "ended": meta["ended"], "elapsed": jobs.seconds(meta["started"], meta["ended"]),
                         "stages": jobs.stages(meta), "log": jobs.log_tail(job_id)})


def job_log(request, job_id: str):
    """A job's whole log as plain text, for copying."""
    if jobs.status(job_id) is None:
        raise Http404
    return HttpResponse(jobs.log_tail(job_id, lines=None), content_type="text/plain; charset=utf-8")


def activity(request):
    """GOPACS requests and notices over the last 30 days."""
    context = _common(request, "activity")
    found = data.activity(30)
    if found:
        daily = found["daily"]
        feed = daily.get("feed-in", pd.Series(0, index=daily.index)).tolist()
        off = daily.get("offtake", pd.Series(0, index=daily.index)).tolist()
        labels = [d.strftime("%d") for d in daily.index]
        tips = [f"{d:%a %d %b}|Feed-in {f} areas · Offtake {o} areas" for d, f, o in zip(daily.index, feed, off)]
        found["chart"] = charts.bars(labels, [("Feed-in", "#41609e", feed), ("Offtake", "#c9821b", off)], tips,
                                     width=1340, height=260)
        found["busiest"] = found["busiest"].to_dict("records")
        found["operators"] = found["operators"].to_dict("records")
        top = max((o["area_days"] for o in found["operators"]), default=1)
        for o in found["operators"]:
            o["share"] = o["area_days"] / top * 100
    context.update(found=found, notices=data.notices(8).to_dict("records"))
    return render(request, "console/activity.html", context)


def performance(request):
    """How often flags were right: the 14-day test, the live ledger and the validation history."""
    context = _common(request, "performance")
    threshold = context["threshold"] or data.CONFIG.model.day_threshold
    rolling = data.rolling_test()
    if rolling:
        days = rolling["days"]
        labels = [d["delivery_date"].strftime("%d") for d in days]
        tips = [f"{d['delivery_date']:%a %d %b}|Shown: {d['shown_correct']} correct · {d['shown_wrong']} wrong · "
                f"{d['requests']} area-days with requests" for d in days]
        context["flag_chart"] = charts.bars(labels, [("Correct", "#41609e", [d["shown_correct"] for d in days]),
                                                     ("Wrong", "#c9821b", [d["shown_wrong"] for d in days])], tips,
                                            width=1340, height=220)
    context.update(
        threshold=threshold,
        track=data.track_record().head(30).to_dict("records"),
        rolling=rolling,
        validation=data.validation_history().head(20).to_dict("records"),
    )
    return render(request, "console/performance.html", context)


def model(request):
    """The trained model, its inspector and the settings."""
    context = _common(request, "model")
    bundle = data.bundle()
    config = data.CONFIG
    if bundle:
        veto = bundle.get("inspector") or {}
        context.update(
            bundle=bundle,
            estimator_name=type(bundle["estimator"]).__name__,
            trained_at=pd.Timestamp(bundle["trained_at"]).tz_convert("Europe/Amsterdam"),
            groups=data.feature_groups(bundle["features"]),
            veto=veto,
            veto_share=veto["trained_correct"] / veto["trained_flags"] if veto else None,
            veto_inputs=len(veto.get("columns", [])),
        )
    context["settings"] = [
        ("History starts", f"{config.grid.start_date:%d %b %Y}"),
        ("Grid cell size", f"{config.grid.cell_degrees}°"),
        ("Directions", ", ".join(config.grid.directions)),
        ("Forecast issued at", f"{config.features.issue_hour}:00 the day before"),
        ("Area active if a request in the last", f"{config.features.active_days} days"),
        ("Day flag threshold", f"{config.model.day_threshold}"),
        ("Inspector veto below", f"{config.inspector.veto_cut:.0%} confidence"),
        ("Inspector learns from", f"{config.inspector.start_date:%b %Y}"),
        ("Validation window", f"{config.validation.window_days} days"),
    ]
    return render(request, "console/model.html", context)
