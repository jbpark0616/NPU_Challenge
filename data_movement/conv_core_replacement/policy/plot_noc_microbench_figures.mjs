import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const DATA = path.join(ROOT, "data_movement", "data", "blackhole");
const OUT = path.join(
  ROOT,
  "data_movement",
  "conv_core_replacement",
  "policy",
  "figures",
  "noc_microbench",
);
fs.mkdirSync(OUT, { recursive: true });

const C = {
  blue: "#4472C4",
  orange: "#ED7D31",
  gray: "#A6A6A6",
  grid: "#D9D9D9",
  ink: "#111111",
};

function csv(name) {
  const rows = fs.readFileSync(path.join(DATA, name), "utf8").trim().split(/\r?\n/);
  const header = rows.shift().split(",");
  return rows.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(header.map((key, i) => [key, values[i]]));
  });
}

function esc(text) {
  return String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function svgFrame(inner, width = 1440, height = 760) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="100%" height="100%" fill="white"/>
  <style>
    text { font-family: "Times New Roman", Times, serif; fill: ${C.ink}; }
    .axis-label { font-size: 34px; font-weight: 700; }
    .tick { font-size: 27px; }
    .legend { font-size: 29px; }
  </style>
  ${inner}
</svg>\n`;
}

function lineChart({ series, labels, yMin = 0, yMax, yStep, xLabel, yLabel, stem }) {
  const W = 1440, H = 760;
  const m = { l: 165, r: 55, t: 90, b: 145 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const sx = (i) => m.l + (labels.length === 1 ? pw / 2 : (i * pw) / (labels.length - 1));
  const sy = (v) => m.t + ph - ((v - yMin) / (yMax - yMin)) * ph;
  const parts = [];

  for (let v = yMin; v <= yMax + 1e-9; v += yStep) {
    const y = sy(v);
    parts.push(`<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="${C.grid}" stroke-width="2"/>`);
    parts.push(`<text class="tick" x="${m.l - 22}" y="${y + 9}" text-anchor="end">${v}</text>`);
  }
  parts.push(`<line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${m.t + ph}" stroke="${C.ink}" stroke-width="3"/>`);
  parts.push(`<line x1="${m.l}" y1="${m.t + ph}" x2="${W - m.r}" y2="${m.t + ph}" stroke="${C.ink}" stroke-width="3"/>`);
  labels.forEach((label, i) => {
    const x = sx(i);
    parts.push(`<line x1="${x}" y1="${m.t + ph}" x2="${x}" y2="${m.t + ph + 13}" stroke="${C.ink}" stroke-width="3"/>`);
    parts.push(`<text class="tick" x="${x}" y="${m.t + ph + 48}" text-anchor="middle">${esc(label)}</text>`);
  });
  parts.push(`<text class="axis-label" x="${m.l + pw / 2}" y="${H - 28}" text-anchor="middle">${esc(xLabel)}</text>`);
  parts.push(`<text class="axis-label" x="42" y="${m.t + ph / 2}" text-anchor="middle" transform="rotate(-90 42 ${m.t + ph / 2})">${esc(yLabel)}</text>`);

  series.forEach((s, si) => {
    const points = s.values.map((v, i) => `${sx(i)},${sy(v)}`).join(" ");
    parts.push(`<polyline points="${points}" fill="none" stroke="${s.color}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>`);
    s.values.forEach((v, i) => {
      const marker = si === 0
        ? `<circle cx="${sx(i)}" cy="${sy(v)}" r="9" fill="${s.color}"/>`
        : `<rect x="${sx(i) - 8}" y="${sy(v) - 8}" width="16" height="16" fill="${s.color}"/>`;
      parts.push(marker);
    });
  });

  const legendX = m.l + 22, legendY = 46;
  series.forEach((s, i) => {
    const x = legendX + i * 170;
    parts.push(`<line x1="${x}" y1="${legendY}" x2="${x + 52}" y2="${legendY}" stroke="${s.color}" stroke-width="6"/>`);
    parts.push(`<text class="legend" x="${x + 68}" y="${legendY + 10}">${esc(s.name)}</text>`);
  });
  fs.writeFileSync(path.join(OUT, `${stem}.svg`), svgFrame(parts.join("\n")));
}

function barRangeChart({ rows, metric, yMax, yStep, yLabel, stem }) {
  const W = 1560, H = 760;
  const m = { l: 165, r: 55, t: 100, b: 150 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const step = pw / rows.length;
  const barW = step * 0.58;
  const sy = (v) => m.t + ph - (v / yMax) * ph;
  const labels = ["Full", "Row 0", "Row 1", "Row 2", "Row 3", "Col 0", "Col 1", "Col 2", "Col 3", "Col 4"];
  const parts = [];

  for (let v = 0; v <= yMax + 1e-9; v += yStep) {
    const y = sy(v);
    parts.push(`<line x1="${m.l}" y1="${y}" x2="${W - m.r}" y2="${y}" stroke="${C.grid}" stroke-width="2"/>`);
    parts.push(`<text class="tick" x="${m.l - 22}" y="${y + 9}" text-anchor="end">${v}</text>`);
  }
  parts.push(`<line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${m.t + ph}" stroke="${C.ink}" stroke-width="3"/>`);
  parts.push(`<line x1="${m.l}" y1="${m.t + ph}" x2="${W - m.r}" y2="${m.t + ph}" stroke="${C.ink}" stroke-width="3"/>`);

  rows.forEach((row, i) => {
    const x = m.l + step * (i + 0.5);
    const mean = Number(row[`${metric}_mean`]);
    const min = Number(row[`${metric}_min`]);
    const max = Number(row[`${metric}_max`]);
    const color = Number(row.segments) === 1 ? C.blue : C.orange;
    parts.push(`<rect x="${x - barW / 2}" y="${sy(mean)}" width="${barW}" height="${m.t + ph - sy(mean)}" fill="${color}" stroke="${C.ink}" stroke-width="1.4"/>`);
    parts.push(`<line x1="${x}" y1="${sy(max)}" x2="${x}" y2="${sy(min)}" stroke="${C.ink}" stroke-width="2"/>`);
    parts.push(`<line x1="${x - 9}" y1="${sy(max)}" x2="${x + 9}" y2="${sy(max)}" stroke="${C.ink}" stroke-width="2"/>`);
    parts.push(`<line x1="${x - 9}" y1="${sy(min)}" x2="${x + 9}" y2="${sy(min)}" stroke="${C.ink}" stroke-width="2"/>`);
    parts.push(`<text class="tick" x="${x}" y="${m.t + ph + 45}" text-anchor="middle">${labels[i]}</text>`);
  });

  parts.push(`<text class="axis-label" x="${m.l + pw / 2}" y="${H - 28}" text-anchor="middle">Excluded row or column</text>`);
  parts.push(`<text class="axis-label" x="42" y="${m.t + ph / 2}" text-anchor="middle" transform="rotate(-90 42 ${m.t + ph / 2})">${esc(yLabel)}</text>`);
  const legend = [
    [C.blue, "1 rectangle"],
    [C.orange, "2 rectangles"],
  ];
  legend.forEach(([color, label], i) => {
    const x = m.l + 20 + i * 245;
    parts.push(`<rect x="${x}" y="35" width="27" height="27" fill="${color}" stroke="${C.ink}" stroke-width="1"/>`);
    parts.push(`<text class="legend" x="${x + 42}" y="58">${label}</text>`);
  });
  fs.writeFileSync(path.join(OUT, `${stem}.svg`), svgFrame(parts.join("\n"), W, H));
}

const arrivalLines = fs
  .readFileSync(path.join(DATA, "arrival", "arrival_per_core.csv"), "utf8")
  .trim()
  .split(/\r?\n/)
  .slice(1);
const arrivalByRun = new Map();
for (const line of arrivalLines) {
  const match = line.match(/^(\d+),".*",\d+,\d+,([0-9.]+)$/);
  if (!match) continue;
  const run = Number(match[1]);
  if (!arrivalByRun.has(run)) arrivalByRun.set(run, []);
  arrivalByRun.get(run).push(Number(match[2]));
}
function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
const senderPositions = Array.from({ length: 20 }, (_, pos) => `${pos % 5},${Math.floor(pos / 5)}`);
lineChart({
  labels: senderPositions,
  series: [0, 1].map((noc, i) => ({
    name: `NoC${noc}`,
    color: i === 0 ? C.blue : C.orange,
    values: Array.from({ length: 20 }, (_, pos) => median(arrivalByRun.get((noc * 20 + pos) * 1024)) / 1000),
  })),
  yMin: 60,
  yMax: 80,
  yStep: 5,
  xLabel: "Sender position (x,y)",
  yLabel: "Median receiver wait (×10³ cycles)",
  stem: "sender_position_latency",
});

const shapeRows = csv("Multicast Schemes Single.csv");
const shapes = ["3 x 3", "5 x 2", "4 x 3", "3 x 4", "4 x 4", "5 x 4"];
const shapeLabels = ["3×3 (9)", "5×2 (10)", "4×3 (12)", "3×4 (12)", "4×4 (16)", "5×4 (20)"];
lineChart({
  labels: shapeLabels,
  series: [0, 1].map((noc, i) => ({
    name: `NoC${noc}`,
    color: i === 0 ? C.blue : C.orange,
    values: shapes.map((shape) => Number(shapeRows.find((r) => Number(r["NoC Index"]) === noc && r["Grid Dimensions"] === shape)["Latency (cycles)"]) / 1000),
  })),
  yMin: 60,
  yMax: 80,
  yStep: 5,
  xLabel: "Multicast rectangle (core count)",
  yLabel: "Latency (×10³ cycles)",
  stem: "rectangle_shape_noc_latency",
});

const segmented = csv("segmented_mcast_latency_summary.csv");
barRangeChart({ rows: segmented, metric: "sender", yMax: 850, yStep: 200, yLabel: "Sender latency (cycles)", stem: "segmented_sender_latency" });
barRangeChart({ rows: segmented, metric: "receiver", yMax: 1000, yStep: 200, yLabel: "Receiver latency (cycles)", stem: "segmented_receiver_latency" });

console.log(`Wrote four SVG figures to ${OUT}`);
