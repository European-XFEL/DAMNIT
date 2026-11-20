"use strict";

const el = id => document.getElementById(id);
const numbers = new Intl.NumberFormat(undefined, {maximumFractionDigits: 2});
const timeFormat = new Intl.DateTimeFormat(undefined, {
  year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  hourCycle: "h23", timeZone: "UTC"
});
const timings = [
  ["Pending", "pending", "#155aa4"],
  ["File merge", "file", "#976019"],
  ["DB write", "database", "#13766b"],
];
const pageSize = 50;
let rows = [];
let page = 0;
let request;

function duration(seconds) {
  if (seconds === null) return "—";
  return seconds > 0 && seconds < 1
    ? numbers.format(seconds * 1000) + " ms"
    : numbers.format(seconds) + " s";
}

function samples(count) {
  return numbers.format(count) + (count === 1 ? " sample" : " samples");
}

function element(tag, text) {
  const node = document.createElement(tag);
  node.textContent = text;
  return node;
}

function metric(label, value, detail) {
  const group = document.createElement("div");
  group.append(element("dt", label), element("dd", value));
  if (detail !== undefined) {
    const sample = element("dd", detail);
    sample.className = "sample";
    group.append(sample);
  }
  return group;
}

function plot(id, traces, layout, emptyText) {
  const node = el(id);
  node.classList.toggle("empty", traces.length === 0);
  if (!traces.length) {
    if (node.data) Plotly.purge(node);
    node.textContent = emptyText;
    return;
  }
  if (!node.data) node.replaceChildren();
  return Plotly.react(node, traces, {
    autosize: true, margin: {l: 50, r: 12, t: 4, b: 38},
    font: {family: 'Inter, "Helvetica Neue", Arial, sans-serif', color: "#405669"},
    xaxis: {type: "date", showgrid: false, nticks: 5, automargin: true},
    showlegend: true, legend: {orientation: "h", y: 1.02, yanchor: "bottom"},
    hovermode: "x unified",
    ...layout
  }, {responsive: true, displayModeBar: false});
}

function drawCharts(data) {
  const {summary, series} = data;
  const x = series.map(bucket => bucket.time);
  const axes = {rangemode: "tozero", gridcolor: "#e3eaf0", zerolinecolor: "#bccad5", automargin: true};
  const mergeTraces = timings.filter(([, field]) => summary[field + "_samples"] > 0)
    .map(([name, field, color]) => ({
      name, type: "scatter", mode: "lines+markers", x,
      y: series.map(bucket => bucket[field + "_seconds"]),
      customdata: series.map(bucket => bucket[field + "_samples"]),
      line: {color, width: 2}, marker: {size: 5}, connectgaps: false,
      hovertemplate: "%{y:.4~g} s · %{customdata} samples<extra></extra>"
    }));
  const jobTraces = [
    ["Raw", "raw_submissions", "#155aa4"],
    ["Processed", "proc_submissions", "#13877d"],
    ["Unclassified", "unclassified_submissions", "#697789"]
  ].filter(([, field]) => summary[field] > 0).map(([name, field, color]) => ({
    name, type: "bar", x, y: series.map(bucket => bucket[field]),
    marker: {color}, hovertemplate: "%{y:d} jobs<extra></extra>"
  }));
  return Promise.all([
    plot("combiner-chart", mergeTraces, {yaxis: {...axes, title: {text: "Seconds"}}},
      "No measured timings in this selection."),
    plot("listener-chart", jobTraces, {barmode: "stack", yaxis: {
      ...axes, title: {text: "Jobs"}, tickformat: ",d",
      dtick: series.every(bucket => bucket.submissions <= 5) ? 1 : undefined
    }},
      "No Slurm submissions in this selection.")
  ]);
}

function drawSummary(s) {
  el("workload").replaceChildren(
    metric("Proposals", numbers.format(s.proposal_count)),
    metric("Runs", numbers.format(s.run_count)),
    metric("Fragments", numbers.format(s.fragments)),
    metric("Variables reported", numbers.format(s.variables), s.variable_samples
      ? numbers.format(s.variables / s.variable_samples) + " per fragment · " + samples(s.variable_samples)
      : "No variable counts")
  );
  el("timings").replaceChildren(...timings.map(([label, field]) =>
    metric("Mean " + (field === "database" ? label : label.toLowerCase()), duration(s[field + "_seconds"]), samples(s[field + "_samples"]))));
  el("listener-metrics").replaceChildren(
    metric("Jobs submitted", numbers.format(s.submissions)),
    metric("Processing failures", numbers.format(s.listener_failures))
  );
  el("combiner-note").textContent = numbers.format(s.renames) + " first "
    + (s.renames === 1 ? "file" : "files") + " created (untimed) · "
    + numbers.format(s.skipped) + " missing " + (s.skipped === 1 ? "fragment" : "fragments") + " skipped";
}

function filterButton(value, title, select) {
  const button = element("button", value);
  button.className = "link";
  button.setAttribute("aria-label", title);
  button.addEventListener("click", () => {
    select();
    el("filters").requestSubmit();
    el("proposal").focus();
  });
  return button;
}

function drawTable() {
  const body = el("run-rows");
  body.replaceChildren();
  const first = page * pageSize;
  for (const item of rows.slice(first, first + pageSize)) {
    const tr = body.insertRow();
    tr.insertCell().append(filterButton("p" + item.proposal, "Filter by proposal " + item.proposal, () => {
      el("proposal").value = item.proposal;
      el("run").value = "";
    }));
    tr.insertCell().append(filterButton("r" + item.run, "Filter by proposal " + item.proposal + ", run " + item.run, () => {
      el("proposal").value = item.proposal;
      el("run").value = item.run;
    }));
    for (const key of ["submissions", "fragments", "variables"]) {
      tr.insertCell().textContent = numbers.format(item[key]);
    }
    for (const [, key] of timings) {
      const cell = tr.insertCell();
      cell.textContent = duration(item[key + "_seconds"]);
      if (item[key + "_samples"]) {
        const count = element("span", samples(item[key + "_samples"]));
        count.className = "sample";
        cell.append(count);
      }
    }
  }
  if (!rows.length) {
    const cell = body.insertRow().insertCell();
    cell.colSpan = 8;
    cell.textContent = "No proposal and run pairs in this selection.";
  }
  el("pagination").hidden = rows.length <= pageSize;
  el("previous").disabled = page === 0;
  el("next").disabled = first + pageSize >= rows.length;
  el("page-info").textContent = "Page " + (page + 1) + " of " + Math.ceil(rows.length / pageSize);
}

function setCustomRange(start, end) {
  el("start").value = start.toISOString().slice(0, 16);
  el("end").value = end.toISOString().slice(0, 16);
}

async function refresh() {
  let start, end;
  if (el("period").value === "custom") {
    start = new Date(el("start").value + "Z");
    end = new Date(el("end").value + "Z");
    if (!(start < end)) {
      el("end").setCustomValidity("Choose an end time after the start time.");
      el("end").reportValidity();
      return;
    }
  } else {
    end = new Date();
    start = new Date(end - Number(el("period").value) * 3600000);
    setCustomRange(start, end);
  }

  const params = new URLSearchParams({start: start.toISOString(), end: end.toISOString()});
  for (const id of ["proposal", "run"]) {
    if (el(id).value !== "") params.set(id, el(id).value);
  }
  if (request) request.abort();
  const controller = new AbortController();
  request = controller;
  el("clear").hidden = !params.has("proposal") && !params.has("run");
  el("status").classList.remove("error");
  el("status").textContent = "Reading service journal…";
  el("results").hidden = true;
  try {
    const response = await fetch("/api/stats?" + params, {signal: controller.signal});
    const data = await response.json();
    if (!response.ok) {
      const message = typeof data.detail === "string" ? data.detail : "Check the selected time range and filters.";
      throw new Error(message);
    }
    if (controller.signal.aborted) return;
    const interval = timeFormat.format(new Date(data.start)) + " – " + timeFormat.format(new Date(data.end)) + " UTC";
    const length = data.bucket_seconds >= 3600
      ? numbers.format(data.bucket_seconds / 3600) + " h" : numbers.format(data.bucket_seconds / 60) + " min";
    const s = data.summary;
    const hasData = s.run_count || s.submissions || s.listener_failures || s.fragments
      || s.skipped || s.renames || s.variable_samples || s.file_samples || s.database_samples || s.pending_samples;
    el("status").textContent = (hasData ? "" : "No matching service activity. ") + interval + " · " + length + " intervals";
    rows = data.runs;
    page = 0;
    drawSummary(s);
    drawTable();
    el("results").hidden = false;
    await drawCharts(data);
  } catch (error) {
    if (controller.signal.aborted) return;
    el("results").hidden = true;
    el("status").classList.add("error");
    el("status").textContent = "Could not load service statistics. " + error.message;
  }
}

el("filters").addEventListener("submit", event => {
  event.preventDefault();
  refresh();
});
el("period").addEventListener("change", () => {
  const custom = el("period").value === "custom";
  el("end").setCustomValidity("");
  document.querySelectorAll(".custom-date").forEach(label => {
    label.hidden = !custom;
    label.querySelector("input").disabled = !custom;
  });
  if (custom) el("start").focus();
  else el("filters").requestSubmit();
});
for (const id of ["start", "end"]) {
  el(id).addEventListener("input", () => el("end").setCustomValidity(""));
}
el("clear").addEventListener("click", () => {
  el("proposal").value = el("run").value = "";
  el("filters").requestSubmit();
  el("proposal").focus();
});
el("previous").addEventListener("click", () => { page--; drawTable(); });
el("next").addEventListener("click", () => { page++; drawTable(); });
refresh();
