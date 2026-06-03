from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from python.utils.snapshots import load_snapshots


def _decimated_indices(count: int, max_particles: int) -> np.ndarray:
    if count <= max_particles:
        return np.arange(count)
    return np.linspace(0, count - 1, max_particles, dtype=np.int64)


def _scene_bounds(snapshots) -> tuple[list[float], float]:
    positions = np.vstack([snapshot.positions for snapshot in snapshots])
    center = positions.mean(axis=0)
    radius = float(np.max(np.linalg.norm(positions - center, axis=1)))
    return center.tolist(), max(radius, 1.0e-12)


def _viewer_payload(snapshots, max_particles: int) -> dict:
    center, radius = _scene_bounds(snapshots)
    frames = []
    for snapshot in snapshots:
        indices = _decimated_indices(len(snapshot.positions), max_particles)
        points = np.column_stack(
            [
                snapshot.positions[indices],
                snapshot.group_id[indices],
                snapshot.masses[indices],
            ]
        )
        frames.append(
            {
                "step": int(snapshot.step),
                "time": float(snapshot.time),
                "points": np.round(points, 8).tolist(),
            }
        )
    return {"center": center, "radius": radius, "frames": frames}


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FMM Galaxy Viewer</title>
<style>
html, body { margin: 0; height: 100%; background: #05070a; color: #e8edf2; font-family: system-ui, sans-serif; }
#wrap { display: grid; grid-template-rows: 1fr auto; height: 100%; }
canvas { width: 100%; height: 100%; display: block; background: radial-gradient(circle at 50% 50%, #111827 0%, #05070a 70%); }
#hud { position: absolute; top: 14px; left: 14px; padding: 10px 12px; background: rgba(5,7,10,0.72); border: 1px solid rgba(255,255,255,0.14); border-radius: 8px; backdrop-filter: blur(8px); }
#controls { display: grid; grid-template-columns: auto minmax(180px, 1fr) repeat(5, auto); gap: 10px; align-items: center; padding: 10px; background: #0b1118; border-top: 1px solid #263241; }
button, select, input { accent-color: #67e8f9; }
button, select { background: #121b26; color: #e8edf2; border: 1px solid #344255; border-radius: 6px; padding: 6px 8px; }
label { color: #aab6c5; font-size: 13px; }
#frame { width: 100%; }
</style>
</head>
<body>
<div id="wrap">
  <canvas id="view"></canvas>
  <div id="hud">
    <div><strong>FMM Galaxy Viewer</strong></div>
    <div id="meta"></div>
    <div id="hint">drag rotate, wheel zoom</div>
  </div>
  <div id="controls">
    <button id="play">Play</button>
    <input id="frame" type="range" min="0" max="0" value="0">
    <label>Projection <select id="projection"><option value="free">3D</option><option value="xy">XY</option><option value="xz">XZ</option><option value="yz">YZ</option></select></label>
    <label>Size <input id="size" type="range" min="1" max="8" value="2"></label>
    <label>Speed <input id="speed" type="range" min="1" max="12" value="4"></label>
    <button id="reset">Reset</button>
  </div>
</div>
<script>
const scene = __SCENE_JSON__;
const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const meta = document.getElementById("meta");
const frameSlider = document.getElementById("frame");
const playButton = document.getElementById("play");
const projectionSelect = document.getElementById("projection");
const sizeSlider = document.getElementById("size");
const speedSlider = document.getElementById("speed");
const resetButton = document.getElementById("reset");
const colors = ["#67e8f9", "#f472b6", "#a7f3d0", "#facc15", "#c084fc", "#fb7185", "#93c5fd", "#fdba74"];
let frame = 0;
let playing = false;
let yaw = 0.7;
let pitch = 0.35;
let zoom = 0.42;
let dragging = false;
let lastX = 0;
let lastY = 0;
let lastTick = 0;
frameSlider.max = Math.max(0, scene.frames.length - 1);
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
function projectPoint(point) {
  const cx = scene.center[0], cy = scene.center[1], cz = scene.center[2];
  let x = (point[0] - cx) / scene.radius;
  let y = (point[1] - cy) / scene.radius;
  let z = (point[2] - cz) / scene.radius;
  const projection = projectionSelect.value;
  let px = x, py = y, depth = z;
  if (projection === "xz") { py = z; depth = y; }
  else if (projection === "yz") { px = y; py = z; depth = x; }
  else if (projection === "free") {
    const cyaw = Math.cos(yaw), syaw = Math.sin(yaw);
    const cpitch = Math.cos(pitch), spitch = Math.sin(pitch);
    const x1 = cyaw * x - syaw * z;
    const z1 = syaw * x + cyaw * z;
    const y1 = cpitch * y - spitch * z1;
    const z2 = spitch * y + cpitch * z1;
    px = x1; py = y1; depth = z2;
  }
  const scale = Math.min(canvas.clientWidth, canvas.clientHeight) * zoom;
  return [canvas.clientWidth * 0.5 + px * scale, canvas.clientHeight * 0.5 - py * scale, depth];
}
function draw() {
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  const current = scene.frames[frame];
  if (!current) return;
  const projected = current.points.map(p => {
    const screen = projectPoint(p);
    return { x: screen[0], y: screen[1], depth: screen[2], group: p[3], mass: p[4] };
  }).sort((a, b) => a.depth - b.depth);
  const size = Number(sizeSlider.value);
  for (const p of projected) {
    const color = colors[Math.abs(Math.floor(p.group)) % colors.length];
    const alpha = projectionSelect.value === "free" ? Math.max(0.35, Math.min(1.0, 0.75 + 0.25 * p.depth)) : 0.82;
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  meta.textContent = `frame ${frame + 1}/${scene.frames.length} | step ${current.step} | t=${current.time.toFixed(3)} | particles ${current.points.length}`;
  frameSlider.value = String(frame);
}
function tick(time) {
  if (playing && time - lastTick > 1000 / Number(speedSlider.value)) {
    frame = (frame + 1) % scene.frames.length;
    lastTick = time;
    draw();
  }
  requestAnimationFrame(tick);
}
canvas.addEventListener("pointerdown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
canvas.addEventListener("pointerup", () => { dragging = false; });
canvas.addEventListener("pointerleave", () => { dragging = false; });
canvas.addEventListener("pointermove", e => {
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + (e.clientY - lastY) * 0.008));
  lastX = e.clientX; lastY = e.clientY;
  draw();
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  zoom = Math.max(0.08, Math.min(2.5, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
  draw();
}, { passive: false });
frameSlider.addEventListener("input", () => { frame = Number(frameSlider.value); draw(); });
playButton.addEventListener("click", () => { playing = !playing; playButton.textContent = playing ? "Pause" : "Play"; });
projectionSelect.addEventListener("change", draw);
sizeSlider.addEventListener("input", draw);
resetButton.addEventListener("click", () => { yaw = 0.7; pitch = 0.35; zoom = 0.42; draw(); });
window.addEventListener("resize", resize);
resize();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a self-contained interactive HTML galaxy viewer.")
    parser.add_argument("--input", default="experiments/validation/smoke_test")
    parser.add_argument("--output", default="interactive_galaxy_viewer.html")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-particles", type=int, default=8000)
    args = parser.parse_args()

    snapshots = load_snapshots(args.input, stride=args.stride)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {args.input}")

    payload = _viewer_payload(snapshots, args.max_particles)
    html = HTML_TEMPLATE.replace("__SCENE_JSON__", json.dumps(payload, separators=(",", ":")))
    Path(args.output).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
