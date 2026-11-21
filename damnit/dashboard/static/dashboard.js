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
let statsKey;
let lastStatsAt;
let statsError = "";
let paused = false;
let evidence;
let evidenceOrigin;
let lastPoll = Date.now();
const pollMs = 30000;

function utc(value) {
  return new Date(value).toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

function age(value) {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000));
  if (seconds < 60) return seconds + " s ago";
  if (seconds < 3600) return Math.floor(seconds / 60) + " min ago";
  if (seconds < 86400) return Math.floor(seconds / 3600) + " h ago";
  return Math.floor(seconds / 86400) + " d ago";
}

function liveEnabled() {
  return !paused && !evidence && el("period").value !== "custom" && !document.hidden;
}

function showLiveState() {
  const custom = el("period").value === "custom";
  el("pause-live").disabled = custom || Boolean(evidence);
  el("pause-live").textContent = paused ? "Resume" : "Pause";
  el("live-state").textContent = evidence ? "Paused while viewing journal evidence" :
    custom ? "Historical range · updates paused" : paused ? "Updates paused" : "Updates every 30 s";
}

function poll() {
  if (!liveEnabled() || Date.now() - lastPoll < pollMs) return;
  lastPoll = Date.now();
  refreshHealth();
  if (!request && el("filters").checkValidity()) refresh(true);
  if (!logRequest && !searches[service]) loadMessages(true);
}

function openEvidence(start, end, label, origin) {
  if (request) request.abort();
  const filters = Object.fromEntries(["period", "start", "end", "proposal", "run", "pending-threshold"]
    .map(id => [id, el(id).value]));
  evidence = {start: start.toISOString(), end: end.toISOString(), label, filters};
  evidenceOrigin = origin;
  // Show the records around the event even when a prior text search is active.
  searches[service] = "";
  el("log-query").value = "";
  showLiveState();
  loadMessages();
  el("journal-heading").focus();
  el("journal-heading").scrollIntoView({block: "start"});
}

function leaveEvidence(returnToSource) {
  const filters = evidence.filters;
  evidence = null;
  lastPoll = Date.now();
  if (returnToSource) {
    for (const [id, value] of Object.entries(filters)) {
      el(id).value = value;
      el(id).setCustomValidity("");
    }
    showCustomFields();
    if (statsKey !== currentFilterKey()) refresh();
  }
  showLiveState();
  loadMessages();
  if (returnToSource) {
    const origin = evidenceOrigin?.isConnected ? evidenceOrigin : el("attention-heading");
    origin.focus();
    origin.scrollIntoView({block: "center"});
  } else {
    el("log-query").focus();
  }
  evidenceOrigin = null;
}


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

async function drawOperationalChart(data, filtered) {
  const serviceEvents = data.summary.service_operations;
  const counts = {...data.summary.operations, ...serviceEvents};
  const categories = operations.filter(([key]) => counts[key] > 0);
  const values = categories.map(([key]) => data.series.map(bucket => key in serviceEvents ? bucket.service_operations[key] : bucket.operations[key]));
  el("ops-scope").textContent = filtered
    ? "Rows marked service-wide include all runs. Select an interval to inspect its journal."
    : "Row totals cover the selected time range. Select an interval to inspect its journal.";
  const node = el("ops-chart");
  node.style.height = categories.length ? (4 + categories.length * 1.5) + "rem" : "";
  el("ops-legend").hidden = !categories.length;
  const trace = {
    type: "heatmap", x: data.series.map(bucket => bucket.time),
    y: categories.map(([key, label]) => label + (key in serviceEvents ? " (service-wide)" : "") + " · " + numbers.format(counts[key])),
    z: values.map(row => row.map(value => heatScale.filter(([minimum]) => value >= minimum).length - 1)),
    customdata: values, showscale: false, zmin: 0, zmax: heatScale.length - 1,
    colorscale: heatScale.map(([, , color], i) => [i / (heatScale.length - 1), color]),
    ygap: 3,
    hovertemplate: "%{x|%d %b, %H:%M UTC}<br>%{y}<br>%{customdata} in this interval<extra></extra>"
  };
  await plot("ops-chart", categories.length ? [trace] : [], {
    yaxis: {autorange: "reversed", automargin: true, fixedrange: true, type: "category"},
    margin: {l: 10, r: 12, t: 4, b: 38}, hovermode: "closest", showlegend: false
  }, "No operational events in this selection.");
  const intervals = el("event-interval");
  intervals.replaceChildren(element("option", "Choose an interval…"));
  intervals.firstChild.value = "";
  data.series.forEach((bucket, i) => {
    if (values.some(row => row[i] > 0)) {
      const option = element("option", utc(bucket.time));
      option.value = String(i);
      intervals.append(option);
    }
  });
  intervals.parentElement.hidden = !categories.length;
  if (categories.length && currentData === data) {
    node.removeAllListeners("plotly_click");
    node.on("plotly_click", event => {
      const index = event.points[0].pointNumber;
      openBucket(data, Array.isArray(index) ? index[1] : index);
    });
  }
}

function openBucket(data, index) {
  const time = new Date(data.series[index].time).getTime();
  const start = new Date(Math.max(time, new Date(data.start).getTime()));
  const end = new Date(Math.min(time + data.bucket_seconds * 1000, new Date(data.end).getTime()));
  openEvidence(start, end, "Operational events", el("event-interval"));
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
  function count(bucket, field) {
    return field === "listener_failures" ? bucket.listener_failures - bucket.local_failures : bucket[field];
  }
  const jobTraces = [
    ["Observed attempts", "dispatch_attempts", "#57667a"],
    ["Raw Slurm", "raw_submissions", "#155aa4"],
    ["Processed Slurm", "proc_submissions", "#13877d"],
    ["Unclassified Slurm", "unclassified_submissions", "#697789"],
    ["Local fallback requests", "local_fallbacks", "#4d72a5"],
    ["Local failures", "local_failures", "#a43835"], ["Capacity skips", "capacity_skips", "#ad6816"],
    ["Other processing failures", "listener_failures", "#7c486e"]
  ].filter(([, field]) => count(summary, field) > 0).map(([name, field, color]) => ({
    name, type: "bar", x, y: series.map(bucket => count(bucket, field)),
    marker: {color}, hovertemplate: "%{y:d} events<extra></extra>"
  }));
  return plot("listener-chart", jobTraces, {barmode: "group", yaxis: {
    ...yAxis, title: {text: "Events"}, tickformat: ",d",
    dtick: jobTraces.every(trace => trace.y.every(count => count <= 5)) ? 1 : undefined
  }}, "No dispatch events observed in this selection.");
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

function drawAttention(data) {
  const s = data.summary;
  const summary = el("attention-metrics");
  const incidents = data.incidents;
  if (service === "combiner") {
    const pending = data.distributions.pending;
    summary.replaceChildren(
      metric("DB failed after file completion", numbers.format(s.db_after_file_failures)),
      metric("Missing fragments skipped", numbers.format(s.skipped)),
      metric("Median pending", duration(pending.p50)),
      metric("P99 pending", duration(pending.p99)),
      metric("Pending over " + data.pending_threshold_seconds + " s",
        numbers.format(s.pending_exceedances) + " / " + numbers.format(pending.count), "measured starts"));
  } else {
    summary.replaceChildren(
      metric("Observed dispatch attempts", numbers.format(s.dispatch_attempts)),
      metric("Processing failures", numbers.format(s.listener_failures), numbers.format(s.local_failures) + (s.local_failures === 1 ? " local extraction failure" : " local extraction failures")),
      metric("Capacity skips", numbers.format(s.capacity_skips)));
  }
  const groups = [
    ["Failures and skips", incidents.failures, incidents.failure_count],
    ["Slow pending attempts", incidents.slow, incidents.slow_count],
    ["Completion unobserved", incidents.unobserved, incidents.unobserved_count]
  ];
  const host = el("incidents");
  // Keep expanded groups open as live data changes.
  const expanded = new Set(Array.from(host.querySelectorAll("details[open]"), node => node.dataset.group));
  host.replaceChildren();
  for (const [name, list, total] of groups) {
    if (!total) continue;
    const section = document.createElement("section");
    const heading = element("h3", name + " · " + numbers.format(total));
    section.append(heading);
    section.append(incidentTable(list.slice(0, 5)));
    if (list.length > 5) {
      const more = document.createElement("details");
      more.dataset.group = name;
      more.open = expanded.has(name);
      more.append(element("summary", "Show " + (list.length - 5) + " more"), incidentTable(list.slice(5)));
      section.append(more);
    }
    if (total > list.length) {
      section.append(element("p", "Showing " + list.length + " of " + numbers.format(total) + ". Narrow the time range to see more."));
    }
    host.append(section);
  }
  if (!host.childElementCount) host.append(element("p", "No incidents observed in this selection."));
}

function incidentTable(items) {
  const container = document.createElement("div");
  container.className = "table-scroll";
  container.tabIndex = 0;
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const label of ["Observed (UTC)", "Proposal / run", "Stage", "Reason", "Journal"]) {
    const th = element("th", label);
    th.scope = "col";
    head.append(th);
  }
  const body = table.createTBody();
  for (const incident of items) {
    const row = body.insertRow();
    row.insertCell().textContent = utc(incident.time).slice(0, -4);
    const identity = incident.proposal != null && incident.run != null
      ? "p" + incident.proposal + " / r" + incident.run
      : incident.kind === "service_crash" ? "Service-wide" : "Unattributed";
    row.insertCell().textContent = identity + (incident.data_type ? " · " + incident.data_type : "");
    const stage = row.insertCell();
    stage.textContent = incident.stage;
    if (incident.fragment) {
      const file = element("span", incident.fragment);
      file.className = "sample fragment-name";
      stage.append(file);
    }
    row.insertCell().textContent = incident.reason + (incident.seconds != null ? " · " + duration(incident.seconds) : "");
    const button = element("button", "View logs");
    button.className = "link";
    button.type = "button";
    button.setAttribute("aria-label", "View journal around " + utc(incident.time) + " for " + incident.reason);
    button.addEventListener("click", () => {
      const center = new Date(incident.time).getTime();
      const label = [identity, incident.stage, incident.reason].join(" · ");
      openEvidence(new Date(center - 60000), new Date(center + 60000), label, button);
    });
    row.insertCell().append(button);
  }
  container.append(table);
  return container;
}

function drawSummary(s) {
  const population = [
    metric("Proposals", numbers.format(s.proposal_count)),
    metric("Runs", numbers.format(s.run_count))
  ];
  if (service === "listener") {
    el("workload").replaceChildren(...population,
      metric("Slurm submissions", numbers.format(s.submissions)),
      metric("Local fallback requests", numbers.format(s.local_fallbacks))
    );
    return;
  }
  el("workload").replaceChildren(...population,
    metric("Fragments", numbers.format(s.fragments)),
    metric("Variables reported", numbers.format(s.variables), s.variable_samples
      ? numbers.format(s.variables / s.variable_samples) + " per reported update · " + samples(s.variable_samples)
      : "No variable counts")
  );
  el("timings").replaceChildren(...timings.map(([label, field]) =>
    metric("Mean " + (field === "database" ? label : label.toLowerCase()), duration(s[field + "_seconds"]), samples(s[field + "_samples"]))));
  el("combiner-note").textContent = numbers.format(s.renames) + " first "
    + (s.renames === 1 ? "file" : "files") + " created (untimed) · "
    + numbers.format(s.skipped) + " missing " + (s.skipped === 1 ? "fragment" : "fragments") + " skipped · " + numbers.format(s.incomplete)
    + " starts without an observed file completion · "
    + numbers.format(s.unattributed_file_samples) + " unattributed merge timings";
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
  const counts = isCombiner ? [["Fragments", "fragments"], ["Committed writes", "database_samples"], ["Failed after file", "db_after_file_failures"], ["Skipped", "skipped"], ["Variables", "variables"]] : [
    ["Submissions", "submissions"], ["Raw", "raw_submissions"], ["Processed", "proc_submissions"],
    ["Observed attempts", "dispatch_attempts"], ["Local fallback requests", "local_fallbacks"],
    ["Capacity skips", "capacity_skips"], ["All processing failures", "listener_failures"], ["Of which local", "local_failures"]
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

function currentFilterKey() {
  return JSON.stringify([service, el("period").value, el("proposal").value, el("run").value, el("pending-threshold").value,
    ...(el("period").value === "custom" ? [el("start").value, el("end").value] : [])]);
}

async function refresh(background = false) {
  const key = currentFilterKey();
  // Poll only the applied selection while the operator edits the form.
  if (background && statsKey !== key) return;
  const keep = currentData && statsKey === key;
  const threshold = Number(el("pending-threshold").value);
  if (service === "combiner" && !(Number.isFinite(threshold) && threshold > 0)) {
    el("pending-threshold").setCustomValidity("Choose a positive number of seconds.");
    el("pending-threshold").reportValidity();
    return;
  }
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
  if (service === "combiner") params.set("pending_threshold_seconds", String(threshold));
  for (const id of ["proposal", "run"]) {
    if (el(id).value !== "") params.set(id, el(id).value);
  }
  if (request) request.abort();
  const controller = new AbortController();
  request = controller;
  el("clear").hidden = !params.has("proposal") && !params.has("run");
  el("status").classList.remove("error");
  statsError = "";
  if (!keep) {
    currentData = null;
    el("status").textContent = "Loading statistics…";
    el("results").hidden = true;
  }
  statsKey = key;
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
    const hasData = s.run_count || s.dispatch_attempts || s.submissions || s.local_fallbacks || s.capacity_skips || s.listener_failures || s.fragments
      || Object.values(s.operations).some(count => count > 0)
      || s.skipped || s.renames || s.variable_samples || s.file_samples || s.database_samples || s.pending_samples;
    el("status").textContent = (hasData ? "" : "No " + serviceNames[service] + " statistics matched these filters. ")
      + interval + " · " + length + " intervals";
    if (data.data_through && new Date(data.data_through) < new Date(data.end)) {
      el("status").textContent += " · Data through " + timeFormat.format(new Date(data.data_through)) + " UTC";
    }
    currentData = data;
    rows = data.runs;
    if (!background) page = 0;
    page = Math.min(page, Math.max(0, Math.ceil(rows.length / pageSize) - 1));
    lastStatsAt = new Date();
    drawAttention(data);
    drawSummary(s);
    drawTable();
    el("results").hidden = false;
    await drawOperationalChart(data, params.has("proposal") || params.has("run"));
    if (controller.signal.aborted) return;
    if (service === "listener") {
      await drawListenerChart(data);
    } else {
      drawDistributions(data.distributions);
      await Promise.all([drawTimingChart(data), drawThroughputChart(data),
        el("performance").open ? drawScalingChart(data) : null]);
    }
  } catch (error) {
    if (controller.signal.aborted) return;
    statsError = error.message;
    if (!keep) currentData = null;
    el("results").hidden = !keep;
    el("status").classList.add("error");
    showStatsError();
  } finally {
    if (request === controller) request = null;
  }
}

function showStatsError() {
  const message = "Could not refresh service statistics. " + statsError +
    (currentData && lastStatsAt ? " · Showing previous results through " + utc(currentData.end) +
    ", loaded " + age(lastStatsAt) + "." : "");
  if (el("status").textContent !== message) el("status").textContent = message;
}

async function loadMessages(background = false) {
  if (logRequest) logRequest.abort();
  const controller = new AbortController();
  logRequest = controller;
  const query = searches[service];
  const params = new URLSearchParams({service, limit: "100"});
  if (query) params.set("q", query);
  el("evidence-actions").hidden = !evidence;
  if (evidence) {
    params.set("start", evidence.start);
    params.set("end", evidence.end);
    el("journal-scope").textContent = evidence.label + " · " + utc(evidence.start) + " – " + utc(evidence.end)
      + " · All messages for this service in the interval, newest 100 matches.";
  } else {
    el("journal-scope").textContent = "All retained messages for this service. Showing the newest 100 matches, independent of statistics filters.";
  }
  const status = el("log-status");
  status.classList.remove("error");
  if (!background) status.textContent = "Loading messages…";
  el("clear-search").hidden = !query && !el("log-query").value;
  if (!background) {
    el("log-scroll").hidden = true;
    el("log-entries").replaceChildren();
  }
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
    if (!items.length && evidence && data.availability === "outside_retention") {
      status.textContent = "Journal evidence unavailable. This interval predates the retained journal.";
      return;
    }
    if (!items.length && evidence && data.availability === "unknown") {
      status.textContent = "No matching messages. Journal coverage for this interval is unknown.";
      return;
    }
    status.textContent = items.length
      ? name + " · " + items.length + (items.length === 1 ? " message" : " messages") + matching + " · Newest first · UTC"
      : "No " + name + " messages" + matching + (evidence ? " in this interval." : " in the retained journal.");
  } catch (error) {
    if (controller.signal.aborted) return;
    status.classList.add("error");
    status.textContent = "Could not load journal messages. " + error.message;
    if (background && el("log-entries").childElementCount) status.textContent += " · Showing previous messages.";
  } finally {
    if (logRequest === controller) logRequest = null;
  }
}

function selectService(next) {
  if (next === service) return;
  if (request) request.abort();
  currentData = null;
  el("results").hidden = true;
  evidence = evidenceOrigin = null;
  service = next;
  el("performance").hidden = next !== "combiner";
  el("threshold-control").hidden = next !== "combiner";
  el("pending-threshold").disabled = next !== "combiner";
  showLiveState();
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
el("event-interval").addEventListener("change", () => {
  if (el("event-interval").value !== "" && currentData) openBucket(currentData, Number(el("event-interval").value));
});
el("back-to-dashboard").addEventListener("click", () => leaveEvidence(true));
el("all-journal").addEventListener("click", () => leaveEvidence(false));
el("pending-threshold").addEventListener("input", () => el("pending-threshold").setCustomValidity(""));
el("performance").addEventListener("toggle", () => {
  if (el("performance").open && currentData && service === "combiner") drawScalingChart(currentData);
});
el("refresh-health").addEventListener("click", refreshHealth);
el("pause-live").addEventListener("click", () => {
  paused = !paused;
  showLiveState();
  if (!paused) {
    lastPoll = 0;
    poll();
  }
});
document.addEventListener("visibilitychange", poll);
window.setInterval(() => {
  if (document.hidden) return;
  showHealthStatus();
  if (statsError) showStatsError();
  poll();
}, 1000);
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
function showCustomFields() {
  const custom = el("period").value === "custom";
  document.querySelectorAll(".custom-date").forEach(label => {
    label.hidden = !custom;
    label.querySelector("input").disabled = !custom;
  });
}
el("period").addEventListener("change", () => {
  showCustomFields();
  showLiveState();
  el("end").setCustomValidity("");
  if (el("period").value === "custom") el("start").focus();
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
refreshHealth();
refresh();
loadMessages();
