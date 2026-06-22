"""FastAPI application: clip dashboard, clip detail, media, and live SSE stream."""

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


def _query_clips(show: str) -> list[Video]:
    with get_session() as session:
        stmt = select(Video).order_by(Video.received_at.desc())
        if show == "aircraft":
            stmt = stmt.where(Video.has_aircraft == True)  # noqa: E712
        elif show == "none":
            stmt = stmt.where(Video.has_aircraft == False)  # noqa: E712
        elif show == "labelled_aircraft":
            stmt = stmt.where(Video.human_label == "aircraft")
        elif show == "labelled_none":
            stmt = stmt.where(Video.human_label == "none")
        elif show == "labelled":
            stmt = stmt.where(Video.human_label.is_not(None))
        return list(session.exec(stmt).all())


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    clips = _query_clips("labelled_aircraft")
    with get_session() as session:
        recent = list(
            session.exec(select(Video).order_by(Video.received_at.desc()).limit(15)).all()
        )
    return templates.TemplateResponse(
        request, "index.html", {"clips": clips, "recent": recent}
    )


@app.get("/clips", response_class=HTMLResponse)
def clips_partial(request: Request, show: str = Query("aircraft")):
    return templates.TemplateResponse(
        request, "_clip_grid.html", {"clips": _query_clips(show)}
    )


@app.get("/clip/{clip_id}", response_class=HTMLResponse)
def clip_detail(request: Request, clip_id: int):
    with get_session() as session:
        clip = session.get(Video, clip_id)
        if clip is None:
            return Response(status_code=404)
    return templates.TemplateResponse(request, "clip_detail.html", {"clip": clip})


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
