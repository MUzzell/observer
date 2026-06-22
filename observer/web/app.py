"""FastAPI application: dashboard, event views, media, and the live SSE stream."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from sse_starlette.sse import EventSourceResponse

from observer.config import get_settings
from observer.storage.db import Event, Video, VideoStatus, get_session, init_db
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


def _query_events(aircraft: str | None, takeoff_only: bool) -> list[Event]:
    with get_session() as session:
        stmt = select(Event).order_by(Event.created_at.desc())
        if takeoff_only:
            stmt = stmt.where(Event.is_takeoff == True)  # noqa: E712
        if aircraft and aircraft != "all":
            stmt = stmt.where(Event.type == aircraft)
        return list(session.exec(stmt).all())


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    events = _query_events(None, takeoff_only=True)
    with get_session() as session:
        videos = list(
            session.exec(select(Video).order_by(Video.received_at.desc()).limit(15)).all()
        )
    return templates.TemplateResponse(
        request, "index.html", {"events": events, "videos": videos}
    )


@app.get("/events", response_class=HTMLResponse)
def events_partial(
    request: Request,
    aircraft: str = Query("all"),
    takeoff_only: bool = Query(True),
):
    events = _query_events(aircraft, takeoff_only)
    return templates.TemplateResponse(
        request, "_event_grid.html", {"events": events}
    )


@app.get("/event/{event_id}", response_class=HTMLResponse)
def event_detail(request: Request, event_id: int):
    with get_session() as session:
        event = session.get(Event, event_id)
        if event is None:
            return Response(status_code=404)
        video = session.get(Video, event.video_id)
        metrics = json.loads(event.trajectory_json) if event.trajectory_json else {}
    return templates.TemplateResponse(
        request,
        "event_detail.html",
        {"event": event, "video": video, "metrics": metrics},
    )


@app.get("/media/{path:path}")
def media(path: str):
    target = (settings.data_dir / path).resolve()
    # Guard against path traversal outside the data dir.
    if not str(target).startswith(str(settings.data_dir.resolve())):
        return Response(status_code=403)
    if not target.is_file():
        return Response(status_code=404)
    return FileResponse(target)


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
