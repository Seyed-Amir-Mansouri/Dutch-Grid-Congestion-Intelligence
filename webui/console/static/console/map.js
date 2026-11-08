// Live map of the forecast: areas on a real basemap, flagged areas lit by their peak probability.
(function () {
  const holder = document.getElementById("map");
  const source = document.getElementById("map-points");
  if (!holder || !source || !window.L) return;
  const points = JSON.parse(source.textContent);
  const LABELS = { alert: "Flag kept", vetoed: "Vetoed by inspector", clear: "No flag" };
  const SHAPES = {
    alert: '<polygon points="12,3 21.5,19.5 2.5,19.5"/>',
    vetoed: '<polygon points="12,3 21,12 12,21 3,12"/>',
    clear: '<circle cx="12" cy="12" r="5"/>',
  };
  const NL = L.latLngBounds([50.72, 3.3], [53.58, 7.25]);

  const map = L.map(holder, { zoomControl: true, attributionControl: true, minZoom: 7, maxZoom: 16, zoomSnap: 0.25,
                              maxBounds: NL.pad(0.35), maxBoundsViscosity: 0.8 });
  map.fitBounds(NL);
  // PDOK's grey BRT background map: the Dutch government's open base map, no key, from any address
  L.tileLayer("https://service.pdok.nl/brt/achtergrondkaart/wmts/v2_0/grijs/EPSG:3857/{z}/{x}/{y}.png", {
    maxZoom: 16, detectRetina: true, className: "tiles-grey",
    attribution: 'Kaart &copy; <a href="https://www.kadaster.nl">Kadaster</a> / <a href="https://www.pdok.nl">PDOK</a>',
  }).addTo(map);

  const pct = (v) => (v == null ? "–" : Math.round(v * 100) + "%");
  const markers = new Map();
  const order = { clear: 0, vetoed: 1, alert: 2 };
  points.sort((a, b) => order[a.status] - order[b.status] || a.peak - b.peak).forEach((p) => {
    const size = p.status === "clear" ? 14 : 26;
    // brighter, wider light for higher peak probability; a random delay makes alerts flicker out of step
    const style = `--p:${p.peak.toFixed(3)};--delay:${(Math.random() * 2.4).toFixed(2)}s`;
    const icon = L.divIcon({
      className: "",
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
      html: `<div class="lamp lamp-${p.status}" style="${style}"><span class="lamp-halo"></span><span class="lamp-ring"></span>` +
            `<span class="lamp-ring lamp-ring-2"></span><svg viewBox="0 0 24 24">${SHAPES[p.status]}</svg></div>`,
    });
    const baseZ = order[p.status] * 1000 + Math.round(p.peak * 500);
    const marker = L.marker([p.lat, p.lon], { icon, zIndexOffset: baseZ, baseZ,
                                             keyboard: false }).addTo(map);
    let text = `<b>${escapeHtml(p.area)}</b><span class="mt-status mt-${p.status}">${LABELS[p.status]}</span>` +
               `<span>Peak ${pct(p.peak)}${p.inspector != null ? " · inspector " + pct(p.inspector) : ""}</span>`;
    if (p.status !== "clear") text += `<span>Risk window ${escapeHtml(p.window)}</span>`;
    if (p.request != null) text += `<span>${p.request ? "✓ Request came" : "✕ No request"}</span>`;
    marker.bindTooltip(text, { className: "map-tip", direction: "top", offset: [0, -size / 2], opacity: 1 });
    marker.on("click", () => select(p.id, false));
    markers.set(p.id, marker);
  });

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // zoom to the kept alerts, else to any flag, else show the whole country
  function fitAll() {
    const alerts = points.filter((p) => p.status === "alert");
    const flagged = points.filter((p) => p.status !== "clear");
    const focus = (alerts.length ? alerts : flagged).map((p) => [p.lat, p.lon]);
    if (focus.length === 1) map.flyTo(focus[0], 10, { duration: 0.8 });
    else if (focus.length) map.flyToBounds(L.latLngBounds(focus).pad(0.3), { maxZoom: 10, duration: 0.8 });
    else map.fitBounds(NL);
  }

  let selected = null;
  function select(id, fly) {
    if (selected != null) {
      const old = markers.get(selected);
      if (old) {
        old.getElement().querySelector(".lamp").classList.remove("is-selected");
        old.setZIndexOffset(old.options.baseZ);
      }
      document.querySelectorAll(".row-pick.is-selected").forEach((row) => row.classList.remove("is-selected"));
    }
    selected = id;
    const marker = markers.get(id);
    if (!marker) return;
    marker.getElement().querySelector(".lamp").classList.add("is-selected");
    marker.setZIndexOffset(100000);
    if (fly) map.flyTo(marker.getLatLng(), Math.max(map.getZoom(), 10), { duration: 0.9 });
    // only the clicked area keeps its detail box
    markers.forEach((other) => other.closeTooltip());
    marker.openTooltip();
    const row = document.querySelector(`.row-pick[data-point="${id}"]`);
    if (row) {
      row.classList.add("is-selected");
      if (!fly) row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  document.querySelectorAll(".row-pick").forEach((row) => {
    const id = Number(row.dataset.point);
    row.addEventListener("click", () => select(id, true));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select(id, true);
      }
    });
    row.addEventListener("mouseenter", () => {
      const marker = markers.get(id);
      if (marker) marker.getElement().querySelector(".lamp").classList.add("is-hover");
    });
    row.addEventListener("mouseleave", () => {
      const marker = markers.get(id);
      if (marker) marker.getElement().querySelector(".lamp").classList.remove("is-hover");
    });
  });
  document.getElementById("map-reset").addEventListener("click", () => {
    if (selected != null) markers.get(selected).closeTooltip();
    fitAll();
  });
  fitAll();
})();
