from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import plotly
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .journal import JournalError, read_journal
from .stats import summarize


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


@app.get("/api/stats")
def stats(
    start: datetime | None = None,
    end: datetime | None = None,
    proposal: int | None = Query(default=None, ge=0),
    run: int | None = Query(default=None, ge=0),
):
    """Summarize the selected journal interval, defaulting to the past 24 hours."""
    if any(value is not None and value.tzinfo is None for value in (start, end)):
        raise HTTPException(422, "Start and end times must include a time zone.")
    try:
        end = end.astimezone(timezone.utc) if end else datetime.now(timezone.utc)
        start = start.astimezone(timezone.utc) if start else end - timedelta(days=1)
    except OverflowError:
        raise HTTPException(422, "The selected dates are out of range.") from None
    if start.year < 1970 or not timedelta(0) < end - start <= timedelta(days=31):
        raise HTTPException(422, "Choose a range after 1970-01-01 lasting up to 31 days.")
    try:
        with closing(read_journal(start, end)) as records:
            return summarize(records, start, end, proposal=proposal, run=run)
    except JournalError as exc:
        raise HTTPException(503, str(exc)) from exc
