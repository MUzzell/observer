// Live dashboard updates via Server-Sent Events.
(function () {
  const conn = document.getElementById("conn");
  const activity = document.getElementById("activity");

  function log(msg, cls) {
    if (!activity) return;
    const li = document.createElement("li");
    li.className = "act " + (cls || "");
    li.innerHTML = `<span class="t">${new Date().toLocaleTimeString()}</span> ${msg}`;
    activity.prepend(li);
    while (activity.children.length > 40) activity.lastChild.remove();
  }

  // Re-render the timeline, preserving the current aircraft-presence filter.
  function refreshContent() {
    if (window.htmx) {
      htmx.ajax("GET", "/clips" + (location.search || ""),
        { target: "#content", swap: "innerHTML" });
    }
  }

  const es = new EventSource("/stream");
  es.addEventListener("open", () => { conn.textContent = "● live"; conn.className = "conn live"; });
  es.addEventListener("error", () => { conn.textContent = "○ reconnecting"; conn.className = "conn down"; });

  es.addEventListener("video_received", (e) => {
    const d = JSON.parse(e.data);
    log(`received <b>${d.filename}</b>`, "recv");
  });
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    log(`processing <b>${d.filename || d.video_id}</b> ${Math.round(d.progress * 100)}%`);
  });
  es.addEventListener("done", (e) => {
    const d = JSON.parse(e.data);
    if (d.has_aircraft) {
      const t = d.aircraft_type ? ` (${d.aircraft_type})` : "";
      log(`✈ <b>aircraft</b>${t} in ${d.filename} — ${Math.round(d.confidence * 100)}%`, "detect");
    } else {
      log(`no aircraft in ${d.filename}`, "done");
    }
    refreshContent();
  });
})();
