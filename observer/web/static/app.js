// Live dashboard updates via Server-Sent Events.
(function () {
  const conn = document.getElementById("conn");
  const activity = document.getElementById("activity");
  const videos = document.getElementById("videos");

  function log(msg, cls) {
    if (!activity) return;
    const li = document.createElement("li");
    li.className = "act " + (cls || "");
    li.innerHTML = `<span class="t">${new Date().toLocaleTimeString()}</span> ${msg}`;
    activity.prepend(li);
    while (activity.children.length > 40) activity.lastChild.remove();
  }

  function setStatus(id, status, extra) {
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

  function refreshGrid() {
    const form = document.querySelector(".controls form");
    const show = form ? form.querySelector("[name=show]").value : "aircraft";
    if (window.htmx) {
      htmx.ajax("GET", "/clips", { target: "#clip-grid", swap: "innerHTML",
        values: { show } });
    }
  }

  const es = new EventSource("/stream");
  es.addEventListener("open", () => { conn.textContent = "● live"; conn.className = "conn live"; });
  es.addEventListener("error", () => { conn.textContent = "○ reconnecting"; conn.className = "conn down"; });

  es.addEventListener("video_received", (e) => {
    const d = JSON.parse(e.data);
    log(`received <b>${d.filename}</b>`, "recv");
    setStatus(d.video_id, "processing");
  });
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    setStatus(d.video_id, "processing", ` ${Math.round(d.progress * 100)}%`);
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    if (d.has_aircraft) {
      const t = d.aircraft_type ? ` (${d.aircraft_type})` : "";
      log(`✈ <b>aircraft</b>${t} in ${d.filename} — ${Math.round(d.confidence * 100)}%`, "detect");
    } else {
      log(`no aircraft in ${d.filename}`, "done");
    }
    setStatus(d.video_id, "done");
    refreshGrid();
  });
})();
