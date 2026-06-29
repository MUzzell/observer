"""FastAPI application: clip dashboard, clip detail, media, and live SSE stream."""

from __future__ import annotations

import asyncio
import calendar as _calendar
import json
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from itertools import groupby
from pathlib import Path

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, func
from sqlmodel import select
from sse_starlette.sse import EventSourceResponse

from observer.config import get_settings
from observer.storage import files
from observer.storage.db import Video, get_session, init_db
from observer.web.events_bus import EventBus
from observer.worker import WorkerService

settings = get_settings()
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
bus = EventBus()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    worker = WorkerService(bus, settings)
    await worker.start()
    app.state.worker = worker
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="Observer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# Effective time for ordering/grouping: capture time when known, else arrival.
_EFFECTIVE = func.coalesce(Video.captured_at, Video.received_at)


def _effective(v: Video) -> datetime:
    return v.captured_at or v.received_at


# A manual label, when present, OVERRIDES the detector: "aircraft" forces yes,
# "none"/"unreadable" force no. With no manual label we fall back to the detector.
_IS_AIRCRAFT_SQL = case(
    (Video.human_label == "aircraft", True),
    (Video.human_label.in_(("none", "unreadable")), False),
    else_=Video.has_aircraft == True,  # noqa: E712
)


def _is_aircraft(v: Video) -> bool:
    if v.human_label == "aircraft":
        return True
    if v.human_label in ("none", "unreadable"):
        return False
    return v.has_aircraft


def _all_clips(show: str = "all") -> list[Video]:
    with get_session() as session:
        stmt = select(Video).order_by(_EFFECTIVE.desc())
        if show == "aircraft":
            stmt = stmt.where(_IS_AIRCRAFT_SQL)
        elif show == "none":
            stmt = stmt.where(~_IS_AIRCRAFT_SQL)
        return list(session.exec(stmt).all())


def _group_by_day(clips: list[Video]) -> list[dict]:
    """The full event timeline, grouped by calendar day (newest first)."""
    groups = []
    for day, items in groupby(clips, key=lambda v: _effective(v).date()):
        items = list(items)
        groups.append({
            "date": day,
            "clips": items,
            "n_clips": len(items),
            "n_aircraft": sum(1 for v in items if _is_aircraft(v)),
        })
    return groups


def _day_counts() -> dict[date, dict]:
    """Per-day clip / aircraft counts, keyed by date, for the calendar."""
    with get_session() as session:
        rows = session.exec(select(Video)).all()
    counts: dict[date, dict] = {}
    for v in rows:
        agg = counts.setdefault(_effective(v).date(), {"n_clips": 0, "n_aircraft": 0})
        agg["n_clips"] += 1
        if _is_aircraft(v):
            agg["n_aircraft"] += 1
    return counts


def _month_neighbours(first: date) -> tuple[str, str]:
    prev = (first - timedelta(days=1)).replace(day=1)
    nxt = (first + timedelta(days=31)).replace(day=1)
    return prev.strftime("%Y-%m"), nxt.strftime("%Y-%m")


def _build_calendar(month: date, counts: dict[date, dict]) -> list[list[dict]]:
    today = date.today()
    weeks = []
    for week in _calendar.Calendar(firstweekday=0).monthdatescalendar(
            month.year, month.month):
        weeks.append([
            {
                "date": d,
                "in_month": d.month == month.month,
                "n_clips": counts.get(d, {}).get("n_clips", 0),
                "n_aircraft": counts.get(d, {}).get("n_aircraft", 0),
                "today": d == today,
            }
            for d in week
        ])
    return weeks


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, show: str = Query("all"), month: str = Query(None)):
    # The timeline can be filtered by aircraft presence; the calendar is always
    # an unfiltered overview that marks days with detected aircraft.
    counts = _day_counts()
    fallback = max(counts) if counts else date.today()
    try:
        shown_month = datetime.strptime(month, "%Y-%m").date() if month else fallback
    except ValueError:
        shown_month = fallback
    shown_month = shown_month.replace(day=1)
    prev_month, next_month = _month_neighbours(shown_month)
    return templates.TemplateResponse(request, "index.html", {
        "groups": _group_by_day(_all_clips(show)), "show": show,
        "cal_weeks": _build_calendar(shown_month, counts),
        "cal_month": shown_month, "cal_prev": prev_month, "cal_next": next_month,
        "cal_dow": ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"],
        "camera_stream_url": settings.camera_stream_url,
    })


@app.get("/clips", response_class=HTMLResponse)
def clips_partial(request: Request, show: str = Query("all")):
    # Returns the filter bar + timeline so an htmx swap of #content updates both
    # the active filter and the list, without reloading the page (live cam stays).
    return templates.TemplateResponse(
        request, "_timeline.html",
        {"groups": _group_by_day(_all_clips(show)), "show": show})


@app.get("/calendar", response_class=HTMLResponse)
def calendar_partial(request: Request, month: str = Query(None)):
    # Just the calendar, for htmx month navigation — leaves the page (and the
    # live camera stream) untouched.
    counts = _day_counts()
    fallback = max(counts) if counts else date.today()
    try:
        shown_month = datetime.strptime(month, "%Y-%m").date() if month else fallback
    except ValueError:
        shown_month = fallback
    shown_month = shown_month.replace(day=1)
    prev_month, next_month = _month_neighbours(shown_month)
    return templates.TemplateResponse(request, "_calendar.html", {
        "cal_weeks": _build_calendar(shown_month, counts),
        "cal_month": shown_month, "cal_prev": prev_month, "cal_next": next_month,
        "cal_dow": ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"],
    })


@app.get("/clip/{clip_id}", response_class=HTMLResponse)
def clip_detail(request: Request, clip_id: int):
    with get_session() as session:
        clip = session.get(Video, clip_id)
        if clip is None:
            return Response(status_code=404)
    return templates.TemplateResponse(request, "clip_detail.html", {"clip": clip})


@app.get("/clip/{clip_id}/video")
def clip_video(clip_id: int):
    """Serve the clip from wherever it currently lives (incoming/processing/
    processed/source) rather than assuming ``processed/``, so a clip mid-pipeline
    still plays."""
    with get_session() as session:
        clip = session.get(Video, clip_id)
    if clip is None:
        return Response(status_code=404)
    path = files.locate_clip(clip.filename, clip.source_path)
    if path is None:
        return Response(status_code=404)
    return FileResponse(path)


# Manual override labels the dashboard can set; "clear" removes the override.
_LABELS = {"aircraft", "none", "unreadable", "clear"}


@app.post("/clip/{clip_id}/label", response_class=HTMLResponse)
def set_label(request: Request, clip_id: int, value: str = Form(...)):
    if value not in _LABELS:
        return Response(status_code=400)
    with get_session() as session:
        clip = session.get(Video, clip_id)
        if clip is None:
            return Response(status_code=404)
        clip.human_label = None if value == "clear" else value
        session.add(clip)
        session.commit()
        session.refresh(clip)
        return templates.TemplateResponse(request, "_label_box.html", {"clip": clip})


@app.get("/media/{path:path}")
def media(path: str):
    target = (settings.data_dir / path).resolve()
    if not str(target).startswith(str(settings.data_dir.resolve())):
        return Response(status_code=403)
    if not target.is_file():
        return Response(status_code=404)
    return FileResponse(target)


@app.get("/source/{clip_id}")
def source(clip_id: int):
    """Serve an imported clip from its original location (only paths recorded in
    the DB are servable)."""
    with get_session() as session:
        clip = session.get(Video, clip_id)
    if clip is None or not clip.source_path:
        return Response(status_code=404)
    p = Path(clip.source_path)
    if not p.is_file():
        return Response(status_code=404)
    return FileResponse(p)


@app.get("/stream")
async def stream(request: Request):
    queue = bus.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=15)
                    yield {"event": message["type"], "data": json.dumps(message)}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            bus.unsubscribe(queue)

    return EventSourceResponse(event_generator())
