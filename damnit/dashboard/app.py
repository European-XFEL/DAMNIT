from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import plotly
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .cache import CacheError, cached_stats
from .health import get_health
from .journal import JournalError, journal_availability, read_log_entries


STATIC = Path(__file__).parent / "static"
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
app = FastAPI(title="DAMNIT service statistics")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC / "index.html")


@app.get("/assets/plotly.min.js", include_in_schema=False)
def plotly_js():
    return FileResponse(PLOTLY_JS, media_type="text/javascript")


@app.get("/api/health")
def health(response: Response):
    """Current unit, consumer lag and last observed work for both services."""
    response.headers["Cache-Control"] = "no-store"
    return get_health()


def _utc_bounds(start, end):
    if any(value is not None and value.tzinfo is None for value in (start, end)):
        raise HTTPException(422, "Start and end times must include a time zone.")
    try:
        start = start.astimezone(timezone.utc) if start is not None else None
        end = end.astimezone(timezone.utc) if end is not None else None
    except OverflowError:
        raise HTTPException(422, "The selected dates are out of range.") from None
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if ((start is not None and start < epoch) or (end is not None and end <= epoch)
            or (start is not None and end is not None and start >= end)):
        raise HTTPException(422, "Choose an increasing time range after 1970-01-01.")
    return start, end


@app.get("/api/stats")
def stats(
    start: datetime | None = None,
    end: datetime | None = None,
    proposal: int | None = Query(default=None, ge=0),
    run: int | None = Query(default=None, ge=0),
    service: Literal["combiner", "listener"] | None = None,
    pending_threshold_seconds: Annotated[float, Query(gt=0, allow_inf_nan=False)] = 60,
):
    """Summarize the selected journal interval, defaulting to the past 24 hours."""
    start, end = _utc_bounds(start, end)
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(days=1)
    if start.year < 1970 or not timedelta(0) < end - start <= timedelta(days=31):
        raise HTTPException(422, "Choose a range after 1970-01-01 lasting up to 31 days.")
    try:
        return cached_stats(start, end, proposal, run, service=service,
                            pending_threshold_seconds=pending_threshold_seconds)
    except (JournalError, CacheError) as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/logs")
def logs(
    service: Literal["combiner", "listener"],
    response: Response,
    q: str = Query(default="", max_length=256),
    limit: int = Query(default=100, ge=1, le=500),
    start: datetime | None = None,
    end: datetime | None = None,
):
    """Return the newest matching messages from the selected service's retained journal."""
    response.headers["Cache-Control"] = "no-store"
    start, end = _utc_bounds(start, end)
    try:
        if start is None and end is None:
            return {"entries": read_log_entries(service, q, limit)}
        entries = read_log_entries(service, q, limit, start, end)
        return {"entries": entries, "availability": "available" if entries else journal_availability(end)}
    except JournalError as exc:
        raise HTTPException(503, str(exc)) from exc
