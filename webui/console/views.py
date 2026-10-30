from __future__ import annotations

from urllib.parse import urlencode

import pandas as pd
from django.http import Http404, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from . import charts, data, jobs


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
    context.update(
        steps=jobs.steps(),
        entsoe_key=jobs.entsoe_key_set(),
        recent=recent,
        selected=jobs.status(selected) if selected else (recent[0] if recent else None),
        files=data.file_status(),
        error=request.GET.get("error"),
    )
    return render(request, "console/operations.html", context)


@require_POST
def run(request, key: str):
    """Start a step and show its log."""
    titles = {s["key"]: s["title"] for s in jobs.steps()}
    if key not in titles:
        raise Http404
    if key == "entsoe" and not jobs.entsoe_key_set():
        return redirect("/operations?" + urlencode({"error": "Set ENTSOE_API_KEY before starting the web UI"}))
    try:
        meta = jobs.start(key, titles[key], request.POST.dict())
    except (RuntimeError, ValueError, KeyError) as error:
        return redirect("/operations?" + urlencode({"error": str(error) or "Missing form value"}))
    return redirect(f"/operations?job={meta['id']}")


def job(request, job_id: str):
    """A job's state and the end of its log, for live updates."""
    meta = jobs.status(job_id)
    if meta is None:
        raise Http404
    return JsonResponse({"status": meta["status"], "step": meta.get("step"), "steps": len(meta["commands"]),
                         "ended": meta["ended"], "elapsed": jobs.seconds(meta["started"], meta["ended"]),
                         "stages": jobs.stages(meta), "log": jobs.log_tail(job_id)})


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
    """How often flags were right: the live ledger, the walk-forward record and the validation history."""
    context = _common(request, "performance")
    threshold = context["threshold"] or data.CONFIG.model.day_threshold
    monthly = data.monthly_skill(threshold)
    if len(monthly):
        labels = [m.strftime("%b %y") for m in monthly["month"]]
        tips = [f"{m:%B %Y}|{c} correct · {w} wrong · {r} area-days with requests"
                for m, c, w, r in zip(monthly["month"], monthly["correct"], monthly["wrong"], monthly["requests"])]
        context["flag_chart"] = charts.bars(labels, [("Correct", "#41609e", monthly["correct"].tolist()),
                                                     ("Wrong", "#c9821b", monthly["wrong"].tolist())], tips,
                                             width=600, tick_every=3)
        p_tips = [f"{m:%B %Y}|{p:.0%} of flags correct · {r:.0%} of request area-days caught"
                  for m, p, r in zip(monthly["month"], monthly["precision"].fillna(0), monthly["recall"].fillna(0))]
        context["precision_chart"] = charts.line(labels, monthly["precision"].tolist(), p_tips, width=740, tick_every=3)
        recent = monthly.tail(3)
        context["recent"] = {
            "flagged": int(recent["flagged"].sum()), "correct": int(recent["correct"].sum()),
            "requests": int(recent["requests"].sum()),
            "precision": recent["correct"].sum() / max(recent["flagged"].sum(), 1),
            "recall": recent["correct"].sum() / max(recent["requests"].sum(), 1),
            "from": recent["month"].iloc[0], "to": recent["month"].iloc[-1],
        }
        context["monthly"] = monthly.iloc[::-1].to_dict("records")
    context.update(
        threshold=threshold,
        track=data.track_record().head(30).to_dict("records"),
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
