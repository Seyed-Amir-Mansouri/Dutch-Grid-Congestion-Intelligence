// Hover tips for every element with data-tip ("title|details"), and the live process view on the Operations page.
(function () {
  const tip = document.getElementById("tip");

  function show(target, event) {
    const [title, ...rest] = target.dataset.tip.split("|");
    tip.replaceChildren();
    const head = document.createElement("b");
    head.textContent = title;
    tip.append(head);
    if (rest.length) tip.append(rest.join("|"));
    tip.hidden = false;
    move(event);
  }

  function move(event) {
    const pad = 14;
    const box = tip.getBoundingClientRect();
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - pad;
    if (y + box.height > window.innerHeight - 8) y = event.clientY - box.height - pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  document.addEventListener("pointerover", (event) => {
    const target = event.target.closest("[data-tip]");
    if (target && target.dataset.tip) show(target, event);
  });
  document.addEventListener("pointermove", (event) => {
    if (!tip.hidden) move(event);
  });
  document.addEventListener("pointerout", (event) => {
    const target = event.target.closest("[data-tip]");
    if (target && !target.contains(event.relatedTarget)) tip.hidden = true;
  });

  // copy the job's whole log, not only the part shown
  const copy = document.getElementById("copy-log");
  if (copy) {
    copy.addEventListener("click", async () => {
      const label = copy.textContent;
      try {
        const text = await (await fetch(copy.dataset.url, { cache: "no-store" })).text();
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          // opened by IP address over http: the clipboard API is off, so copy through a hidden text box
          const box = document.createElement("textarea");
          box.value = text;
          box.style.position = "fixed";
          box.style.opacity = "0";
          document.body.appendChild(box);
          box.select();
          document.execCommand("copy");
          box.remove();
        }
        copy.textContent = "Copied";
      } catch (error) {
        copy.textContent = "Copy failed";
      }
      setTimeout(() => (copy.textContent = label), 1800);
    });
  }

  const panel = document.getElementById("process");
  if (!panel) return;
  const log = document.getElementById("log");
  const follow = document.getElementById("follow");
  const list = document.getElementById("stages");
  const badge = document.getElementById("job-status");
  const out = {
    stage: document.getElementById("ps-stage"),
    elapsed: document.getElementById("ps-elapsed"),
    left: document.getElementById("ps-left"),
    bar: document.getElementById("ps-bar"),
  };
  const STATE_TEXT = { pending: "Waiting", running: "Running", done: "Done", failed: "Failed", stopped: "Stopped", skipped: "Skipped" };
  const ICONS = {
    GOPACS: '<path d="M4 18h16M7 18V9l5-4 5 4v9M10 18v-4h4v4"/>',
    Weather: '<circle cx="9" cy="9" r="3.5"/><path d="M9 2.5v1.5M2.5 9H4M4.4 4.4l1 1M13.6 4.4l-1 1M8 20h9a3.5 3.5 0 0 0 0-7 5 5 0 0 0-9.6 1.6A2.7 2.7 0 0 0 8 20z"/>',
    Market: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    Locate: '<path d="M12 21s-6-5.6-6-11a6 6 0 0 1 12 0c0 5.4-6 11-6 11z"/><circle cx="12" cy="10" r="2.2"/>',
    Train: '<rect x="6" y="6" width="12" height="12" rx="1.5"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    Forecast: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    Validate: '<path d="M4 12.5l5 5L20 6.5"/>',
    Download: '<path d="M7 17.5h-.5a4.5 4.5 0 0 1-.6-9 6 6 0 0 1 11.6 1.2 3.9 3.9 0 0 1 .5 7.8H17M12 10v10M8.5 16.5 12 20l3.5-3.5"/>',
  };
  let snapshot = null;
  let receivedAt = 0;

  function clock(seconds) {
    if (seconds == null) return "–";
    const s = Math.max(0, Math.round(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const pad = (n) => String(n).padStart(2, "0");
    return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`;
  }

  function build(stages) {
    list.replaceChildren();
    stages.forEach((stage, i) => {
      const item = document.createElement("li");
      item.className = "stage";
      item.innerHTML =
        `<div class="stage-top"><span class="stage-n">${String(i + 1).padStart(2, "0")}</span><span class="stage-state"></span></div>` +
        `<div class="stage-icon"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.7" ` +
        `stroke-linecap="round" stroke-linejoin="round">${ICONS[stage.icon] || ICONS[stage.label] || ICONS.Forecast}</svg><span class="ring"></span></div>` +
        `<h4></h4><p class="stage-detail"></p><div class="stage-bar"><span></span></div><div class="stage-time mono"></div>`;
      item.querySelector("h4").textContent = stage.label;
      item.querySelector(".stage-detail").textContent = stage.detail;
      item.dataset.tip = `${stage.label}|${stage.command}`;
      list.append(item);
    });
  }

  // redraw from the last server snapshot, advancing running times by the local clock
  function draw() {
    if (!snapshot) return;
    const drift = snapshot.status === "running" ? (Date.now() - receivedAt) / 1000 : 0;
    const stages = snapshot.stages;
    if (list.children.length !== stages.length) build(stages);
    let remaining = 0;
    let remainingKnown = true;
    let current = null;
    stages.forEach((stage, i) => {
      const item = list.children[i];
      item.className = "stage stage-" + stage.state;
      item.querySelector(".stage-state").textContent = STATE_TEXT[stage.state] || stage.state;
      const seconds = stage.seconds == null ? null : stage.seconds + (stage.state === "running" ? drift : 0);
      const fill = item.querySelector(".stage-bar span");
      const time = item.querySelector(".stage-time");
      if (stage.state === "running") {
        current = { index: i, stage };
        if (stage.progress != null) {
          // a download reports how much of the file has arrived
          fill.style.width = stage.progress + "%";
          item.classList.remove("stage-unknown");
          if (stage.typical) remaining += Math.max(stage.typical - seconds, 0);
          else remainingKnown = false;
        } else if (stage.icon === "Download") {
          // still starting up, before the first bytes: a moving bar, not a frozen one
          fill.style.width = "";
          item.classList.add("stage-unknown");
          if (stage.typical) remaining += Math.max(stage.typical - seconds, 0);
          else remainingKnown = false;
        } else if (stage.typical) {
          fill.style.width = Math.min(95, (seconds / stage.typical) * 100) + "%";
          item.classList.remove("stage-unknown");
          remaining += Math.max(stage.typical - seconds, 0);
        } else {
          fill.style.width = "";
          item.classList.add("stage-unknown");
          remainingKnown = false;
        }
        time.textContent = stage.progress != null ? `${stage.progress}% · ${clock(seconds)}`
          : stage.icon === "Download" ? `Starting… ${clock(seconds)}`
          : clock(seconds) + (stage.typical ? ` / ~${clock(stage.typical)}` : "");
      } else if (stage.state === "done") {
        fill.style.width = "100%";
        time.textContent = clock(seconds);
      } else if (stage.state === "failed") {
        fill.style.width = "100%";
        time.textContent = clock(seconds) + " · exit ≠ 0";
      } else {
        fill.style.width = "0%";
        time.textContent = stage.typical ? `usually ~${clock(stage.typical)}` : "";
        if (stage.state === "pending") {
          if (stage.typical) remaining += stage.typical;
          else remainingKnown = false;
        }
      }
    });
    // the Files panel row of each file a Drive download is replacing
    stages.forEach((stage) => {
      const parts = stage.command.split(" ");
      if (parts[1] !== "fetch-drive") return;
      const row = document.querySelector(`.file[data-key="${parts[2]}"]`);
      if (!row) return;
      row.classList.toggle("file-loading", stage.state === "running");
      row.classList.toggle("file-starting", stage.state === "running" && stage.progress == null);
      row.classList.toggle("file-queued", stage.state === "pending" && snapshot.status === "running");
      row.classList.toggle("file-fresh", stage.state === "done" && snapshot.status === "running");
      const fill = row.querySelector(".file-load span");
      fill.style.width = stage.state === "running" ? (stage.progress || 0) + "%" : "";
    });
    const done = stages.filter((s) => s.state === "done").length;
    out.bar.style.width = (stages.length ? ((done + (current ? 0.5 : 0)) / stages.length) * 100 : 0) + "%";
    out.bar.parentElement.className = "process-overall overall-" + snapshot.status;
    out.elapsed.textContent = clock(snapshot.elapsed == null ? null : snapshot.elapsed + drift);
    if (snapshot.status === "running") {
      out.stage.textContent = current ? `${current.index + 1} of ${stages.length} · ${current.stage.label}` : "Starting";
      out.left.textContent = remainingKnown ? "~" + clock(remaining) : "learning…";
    } else {
      out.stage.textContent = `${done} of ${stages.length} done`;
      out.left.textContent = "–";
    }
  }

  async function refresh() {
    try {
      const response = await fetch("/operations/job/" + panel.dataset.job, { cache: "no-store" });
      // a failed poll must not end the live view: try again shortly
      if (!response.ok) {
        setTimeout(refresh, 3000);
        return;
      }
      snapshot = await response.json();
      receivedAt = Date.now();
      draw();
      log.textContent = snapshot.log || "Waiting for output…";
      if (follow.checked) log.scrollTop = log.scrollHeight;
      badge.textContent = snapshot.status;
      badge.className = "status status-" + snapshot.status;
      if (snapshot.status === "running") {
        setTimeout(refresh, 1500);
      } else if (panel.dataset.status === "running") {
        // the job just finished: reload so the buttons and recent jobs update
        setTimeout(() => window.location.reload(), 1500);
      }
    } catch (error) {
      setTimeout(refresh, 4000);
    }
  }
  setInterval(draw, 1000);
  refresh();
})();

// The note under an issue-hour choice: after or before the day-ahead market results.
document.addEventListener("change", (event) => {
  const pick = event.target.closest("select.hour-pick");
  if (!pick) return;
  const note = pick.parentElement.querySelector(".field-note");
  if (note) note.textContent = pick.selectedOptions[0].dataset.note;
  // the forecaster tick is only offered before the day-ahead results, and cleared when hidden
  const check = pick.closest("form").querySelector(".check-pick");
  if (check) {
    check.hidden = !("early" in pick.selectedOptions[0].dataset);
    if (check.hidden) check.querySelector("input").checked = false;
  }
});
