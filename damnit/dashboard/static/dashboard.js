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
const operations = [
  ["heartbeat_expired", "Kafka heartbeat expiry"],
  ["offset_commit_failure", "Kafka offset failures"],
  ["extra_deliveries", "Extra deliveries"],
  ["duplicate_source_paths", "Duplicated paths"],
  ["missing_notified_file", "Missing files"],
  ["permission_error", "Permission errors"],
  ["database_locked", "Database locks"],
  ["database_rollback", "DB rollbacks"],
  ["unexpected_processing_error", "Processing errors"],
  ["service_start", "Service starts"],
  ["service_crash", "Service crashes"]
];
const heatScale = [
  [0, "0", "#f0f4f7"], [1, "1", "#aacee9"], [2, "2–4", "#6a9ecd"],
  [5, "5–9", "#3575ae"], [10, "10+", "#174b75"]
];
const yAxis = {rangemode: "tozero", gridcolor: "#e3eaf0", zerolinecolor: "#bccad5", automargin: true};
const pageSize = 50;
const serviceNames = {combiner: "Combiner", listener: "Listener"};
const searches = {combiner: "", listener: ""};
let service = "combiner";
let currentData;
let rows = [];
let page = 0;
let request;
let logRequest;

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

function drawTimingChart(data) {
  const {summary, series} = data;
  const stat = el("trend-stat").value;
  const suffix = stat === "mean" ? "_seconds" : "_" + stat + "_seconds";
  const x = series.map(bucket => bucket.time);
  const traces = timings.filter(([, field]) => summary[field + "_samples"] > 0)
    .map(([name, field, color]) => ({
      name, type: "scatter", mode: "lines+markers", x,
      y: series.map(bucket => bucket[field + suffix]),
      customdata: series.map(bucket => bucket[field + "_samples"]),
      line: {color, width: 2}, marker: {size: 5}, connectgaps: false,
      hovertemplate: "%{y:.4~g} s · %{customdata} samples<extra></extra>"
    }));
  return plot("combiner-chart", traces, {yaxis: {...yAxis, title: {text: "Seconds"}}},
    "No measured timings in this selection.");
}

function drawOperationalChart(data, filtered) {
  const counts = data.summary.operations;
  const categories = operations.filter(([key]) => counts[key] > 0);
  const values = categories.map(([key]) => data.series.map(bucket => bucket.operations[key]));
  el("ops-scope").textContent = filtered
    ? "Only events attributable to the selected proposal or run are shown. Clear the filter to include service-wide events."
    : "Includes service-wide journal records. Row totals cover the selected time range.";
  const node = el("ops-chart");
  node.style.height = categories.length ? (4 + categories.length * 1.5) + "rem" : "";
  el("ops-legend").hidden = !categories.length;
  const trace = {
    type: "heatmap", x: data.series.map(bucket => bucket.time),
    y: categories.map(([key, label]) => label + " · " + numbers.format(counts[key])),
    z: values.map(row => row.map(value => heatScale.filter(([minimum]) => value >= minimum).length - 1)),
    customdata: values, showscale: false, zmin: 0, zmax: heatScale.length - 1,
    colorscale: heatScale.map(([, , color], i) => [i / (heatScale.length - 1), color]),
    ygap: 3,
    hovertemplate: "%{x|%d %b, %H:%M UTC}<br>%{y}<br>%{customdata} in this interval<extra></extra>"
  };
  return plot("ops-chart", categories.length ? [trace] : [], {
    yaxis: {autorange: "reversed", automargin: true, fixedrange: true, type: "category"},
    margin: {l: 10, r: 12, t: 4, b: 38}, hovermode: "closest", showlegend: false
  }, "No operational events in this selection.");
}

function drawScalingChart(data) {
  const {count, points, spearman} = data.scaling;
  const positive = points.length > 0 && points.every(point => point.database_seconds > 0);
  let note = numbers.format(count) + (count === 1 ? " matched update" : " matched updates");
  if (points.length < count) note += " · " + numbers.format(points.length) + " points shown";
  if (spearman !== null) note += " · Spearman ρ " + spearman.toFixed(3);
  el("scaling-note").textContent = note;
  return plot("scaling-chart", points.length ? [{
    type: "scatter", mode: "markers",
    x: points.map(point => point.variables),
    y: points.map(point => point.database_seconds * 1000),
    customdata: points.map(point => [point.proposal, point.run]),
    marker: {size: 6, color: "#155aa4", opacity: 0.45},
    hovertemplate: "p%{customdata[0]} r%{customdata[1]}<br>%{x} variables · %{y:.4~g} ms<extra></extra>"
  }] : [], {
    margin: {l: 60, r: 12, t: 12, b: 48},
    showlegend: false, hovermode: "closest",
    xaxis: {...yAxis, type: "linear", title: {text: "Variables updated"}},
    yaxis: {...yAxis, type: positive ? "log" : "linear", title: {text: positive ? "DB time (ms, log scale)" : "DB time (ms)"}}
  }, "No variable counts matched to successful database writes.");
}

function drawThroughputChart(data) {
  const variables = el("throughput-unit").value === "variables";
  el("throughput-note").textContent = variables ? "Variables reported for database updates in each interval."
    : "Fragment starts include redeliveries. Database writes count successful commits.";
  const fields = variables ? [["Variables", "variables", "#155aa4"]] : [
    ["Fragment starts", "fragments", "#13766b"], ["DB writes", "database_samples", "#976019"]
  ];
  const traces = fields.filter(([, field]) => field === "variables" ? data.summary.variable_samples : data.summary[field])
    .map(([name, field, color]) => ({
      name, type: "scatter", mode: "lines+markers", x: data.series.map(bucket => bucket.time),
      y: data.series.map(bucket => bucket[field]), line: {color, width: 2}, marker: {size: 4},
      hovertemplate: name + " · %{y:d}<extra></extra>"
    }));
  const smallCounts = traces.every(trace => trace.y.every(value => value <= 5));
  return plot("throughput-chart", traces, {
    showlegend: !variables, yaxis: {...yAxis, title: {text: variables ? "Variables" : "Operations"},
      tickformat: ",d", dtick: smallCounts ? 1 : undefined}
  }, variables ? "No reported variable counts in this selection." : "No fragment starts or committed writes in this selection.");
}

function drawListenerChart(data) {
  const {summary, series} = data;
  const x = series.map(bucket => bucket.time);
  const jobTraces = [
    ["Raw", "raw_submissions", "#155aa4"],
    ["Processed", "proc_submissions", "#13877d"],
    ["Unclassified", "unclassified_submissions", "#697789"]
  ].filter(([, field]) => summary[field] > 0).map(([name, field, color]) => ({
    name, type: "bar", x, y: series.map(bucket => bucket[field]),
    marker: {color}, hovertemplate: "%{y:d} jobs<extra></extra>"
  }));
  return plot("listener-chart", jobTraces, {barmode: "stack", yaxis: {
    ...yAxis, title: {text: "Jobs"}, tickformat: ",d",
    dtick: series.every(bucket => bucket.submissions <= 5) ? 1 : undefined
  }}, "No Slurm submissions in this selection.");
}

function drawDistributions(distributions) {
  const body = el("distribution-rows");
  body.replaceChildren();
  for (const [label, key] of timings) {
    const values = distributions[key];
    const row = body.insertRow();
    const heading = element("th", label);
    heading.scope = "row";
    row.append(heading);
    row.insertCell().textContent = numbers.format(values.count);
    for (const field of ["mean", "p50", "p95", "p99", "max"]) {
      row.insertCell().textContent = duration(values[field]);
    }
  }
}

function drawSummary(s) {
  const population = [
    metric("Proposals", numbers.format(s.proposal_count)),
    metric("Runs", numbers.format(s.run_count))
  ];
  if (service === "listener") {
    el("workload").replaceChildren(...population,
      metric("Jobs submitted", numbers.format(s.submissions)),
      metric("Processing failures", numbers.format(s.listener_failures))
    );
    return;
  }
  el("workload").replaceChildren(...population,
    metric("Fragments", numbers.format(s.fragments)),
    metric("Variables reported", numbers.format(s.variables), s.variable_samples
      ? numbers.format(s.variables / s.variable_samples) + " per fragment · " + samples(s.variable_samples)
      : "No variable counts")
  );
  el("timings").replaceChildren(...timings.map(([label, field]) =>
    metric("Mean " + (field === "database" ? label : label.toLowerCase()), duration(s[field + "_seconds"]), samples(s[field + "_samples"]))));
  el("combiner-note").textContent = numbers.format(s.renames) + " first "
    + (s.renames === 1 ? "file" : "files") + " created (untimed) · "
    + numbers.format(s.skipped) + " missing " + (s.skipped === 1 ? "fragment" : "fragments") + " skipped · " + numbers.format(s.incomplete)
    + " starts without an observed file completion";
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
  const isCombiner = service === "combiner";
  const counts = isCombiner ? [["Fragments", "fragments"], ["Variables", "variables"]] : [
    ["Submissions", "submissions"], ["Raw", "raw_submissions"], ["Processed", "proc_submissions"],
    ["Failures", "listener_failures"]
  ];
  const columns = ["Proposal", "Run", ...counts.map(([title]) => title)];
  if (isCombiner) columns.push("Mean pending", "Mean file merge", "Mean DB write");
  el("run-columns").replaceChildren(...columns.map(title => {
    const column = element("th", title);
    column.scope = "col";
    return column;
  }));
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
    for (const [, key] of counts) {
      tr.insertCell().textContent = numbers.format(item[key]);
    }
    if (!isCombiner) continue;
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
    cell.colSpan = columns.length;
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

  const params = new URLSearchParams({service, start: start.toISOString(), end: end.toISOString()});
  for (const id of ["proposal", "run"]) {
    if (el(id).value !== "") params.set(id, el(id).value);
  }
  if (request) request.abort();
  const controller = new AbortController();
  request = controller;
  el("clear").hidden = !params.has("proposal") && !params.has("run");
  el("status").classList.remove("error");
  el("status").textContent = "Loading statistics…";
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
      || Object.values(s.operations).some(count => count > 0)
      || s.skipped || s.renames || s.variable_samples || s.file_samples || s.database_samples || s.pending_samples;
    el("status").textContent = (hasData ? "" : "No " + serviceNames[service] + " statistics matched these filters. ")
      + interval + " · " + length + " intervals";
    if (data.data_through && new Date(data.data_through) < new Date(data.end)) {
      el("status").textContent += " · Data through " + timeFormat.format(new Date(data.data_through)) + " UTC";
    }
    currentData = data;
    rows = data.runs;
    page = 0;
    drawSummary(s);
    drawTable();
    el("results").hidden = false;
    if (service === "listener") {
      await drawListenerChart(data);
    } else {
      drawDistributions(data.distributions);
      await Promise.all([drawTimingChart(data), drawOperationalChart(data, params.has("proposal") || params.has("run")),
        drawScalingChart(data), drawThroughputChart(data)]);
    }
  } catch (error) {
    if (controller.signal.aborted) return;
    el("results").hidden = true;
    el("status").classList.add("error");
    el("status").textContent = "Could not load service statistics. " + error.message;
  }
}

async function loadMessages() {
  if (logRequest) logRequest.abort();
  const controller = new AbortController();
  logRequest = controller;
  const query = searches[service];
  const params = new URLSearchParams({service, limit: "100"});
  if (query) params.set("q", query);
  const status = el("log-status");
  status.classList.remove("error");
  status.textContent = "Loading messages…";
  el("clear-search").hidden = !query && !el("log-query").value;
  el("log-scroll").hidden = true;
  el("log-entries").replaceChildren();
  try {
    const response = await fetch("/api/logs?" + params, {signal: controller.signal});
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : "Check the journal search.");
    }
    if (controller.signal.aborted) return;
    const items = data.entries.map(entry => {
      const item = document.createElement("li");
      const time = element("time", entry.timestamp.replace("T", " ").replace(/(?:Z|\+00:00)$/, ""));
      time.dateTime = entry.timestamp;
      item.append(time, element("pre", entry.message));
      return item;
    });
    el("log-entries").replaceChildren(...items);
    el("log-scroll").hidden = items.length === 0;
    el("log-scroll").scrollTop = 0;
    const name = serviceNames[service];
    const matching = query ? " containing “" + query + "”" : "";
    status.textContent = items.length
      ? name + " · " + items.length + (items.length === 1 ? " message" : " messages") + matching + " · Newest first · UTC"
      : "No " + name + " messages" + matching + " in the retained journal.";
  } catch (error) {
    if (controller.signal.aborted) return;
    status.classList.add("error");
    status.textContent = "Could not load journal messages. " + error.message;
  }
}

function selectService(next) {
  if (next === service) return;
  if (request) request.abort();
  currentData = null;
  el("results").hidden = true;
  service = next;
  for (const name of Object.keys(serviceNames)) {
    const active = name === service;
    const tab = el(name + "-tab");
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    el(name + "-panel").hidden = !active;
    el(name + "-charts").hidden = !active;
  }
  el(service + "-panel").append(el("service-view"));
  el("log-query").value = searches[service];
  loadMessages();
  el("status").classList.remove("error");
  el("status").textContent = "Apply valid statistics filters to load this service.";
  if (el("filters").checkValidity()) refresh();
}

const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectService(tab.id.split("-")[0]));
  tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    tabs[next].focus();
    tabs[next].click();
  });
});
el("log-search").addEventListener("submit", event => {
  event.preventDefault();
  searches[service] = el("log-query").value;
  loadMessages();
});
el("log-query").addEventListener("input", () => {
  el("clear-search").hidden = !el("log-query").value && !searches[service];
});
el("clear-search").addEventListener("click", () => {
  el("log-query").value = "";
  el("log-search").requestSubmit();
  el("log-query").focus();
});
el("refresh-logs").addEventListener("click", () => el("log-search").requestSubmit());

const heatLegend = el("ops-legend");
heatLegend.append(element("span", "Events per interval"));
for (const [, label, color] of heatScale) {
  const group = element("span", label);
  const swatch = document.createElement("i");
  swatch.style.background = color;
  group.prepend(swatch);
  heatLegend.append(group);
}
el("trend-stat").addEventListener("change", () => {
  if (currentData) drawTimingChart(currentData);
});
el("throughput-unit").addEventListener("change", () => {
  if (currentData) drawThroughputChart(currentData);
});
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
loadMessages();
