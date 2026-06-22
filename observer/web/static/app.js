// Live dashboard updates via Server-Sent Events.
(function () {
  const conn = document.getElementById("conn");
  const activity = document.getElementById("activity");
  const videos = document.getElementById("videos");

  function log(msg, cls) {
    if (!activity) return;
    const li = document.createElement("li");
    li.className = "act " + (cls || "");
    const time = new Date().toLocaleTimeString();
    li.innerHTML = `<span class="t">${time}</span> ${msg}`;
    activity.prepend(li);
    while (activity.children.length > 40) activity.lastChild.remove();
  }

  function setVideoStatus(id, status, extra) {
    if (!videos) return;
    let li = videos.querySelector(`[data-video-id="${id}"]`);
    if (!li) {
      li = document.createElement("li");
      li.dataset.videoId = id;
      li.innerHTML = `<span class="fname"></span><span class="status"></span>`;
      videos.prepend(li);
    }
    const badge = li.querySelector(".status");
    badge.textContent = status + (extra || "");
    badge.className = "status status-" + status;
  }

  const es = new EventSource("/stream");
  es.addEventListener("open", () => {
    conn.textContent = "● live";
    conn.className = "conn live";
  });
  es.addEventListener("error", () => {
    conn.textContent = "○ reconnecting";
    conn.className = "conn down";
  });

  es.addEventListener("video_received", (e) => {
    const d = JSON.parse(e.data);
    log(`received <b>${d.filename}</b>`, "recv");
    setVideoStatus(d.video_id, "processing");
  });
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    setVideoStatus(d.video_id, "processing", ` ${Math.round(d.progress * 100)}%`);
  });
  es.addEventListener("event_detected", (e) => {
    const d = JSON.parse(e.data);
    log(`detected <b>${d.aircraft}</b> takeoff (${Math.round(d.confidence * 100)}%) ` +
        `· <a href="/event/${d.event_id}">view</a>`, "detect");
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    log(`finished <b>${d.filename}</b> — ${d.event_count} event(s)`, "done");
    setVideoStatus(d.video_id, "done");
    // Refresh the event grid to include any new detections.
    if (window.htmx) {
      const grid = document.querySelector("#event-grid form, .controls form");
      htmx.ajax("GET", "/events", { target: "#event-grid", swap: "innerHTML",
        values: collectFilters() });
    }
  });
  es.addEventListener("error_event", () => {});
  es.addEventListener("error", () => {});

  function collectFilters() {
    const form = document.querySelector(".controls form");
    if (!form) return {};
    const fd = new FormData(form);
    const out = {};
    for (const [k, v] of fd.entries()) out[k] = v;
    if (!fd.has("takeoff_only")) out["takeoff_only"] = "false";
    return out;
  }
})();
