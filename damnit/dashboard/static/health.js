"use strict";

let healthRequest;
let healthSample;
let healthError = "";
const previousSources = new Map();

function activityStamp(value, activity) {
  if (value) return utc(value);
  if (activity.status !== "ok") return "Unavailable";
  return activity.since ? "None observed since " + utc(activity.since) : "None observed";
}

function drawHealth(data) {
  const cards = [];
  for (const [name, label] of Object.entries(serviceNames)) {
    const {unit, kafka, activity} = data.services[name];
    const card = document.createElement("section");
    const heading = element("h3", label);
    const state = unit.status === "ok"
      ? (unit.load_state === "not-found" ? "Unit not found" : [unit.active_state, unit.sub_state].filter(Boolean).join(" · "))
      : "Unavailable";
    const badge = element("span", state || "Unknown");
    badge.className = "unit-state";
    if (unit.status === "ok" && unit.active_state === "active") badge.classList.add("active");
    if (unit.status === "ok" && ["failed", "inactive"].includes(unit.active_state)) badge.classList.add("stopped");
    heading.append(badge);
    const details = document.createElement("dl");
    const lag = kafka.status === "ok" && kafka.lag !== null ? numbers.format(kafka.lag) + " messages" :
      kafka.status === "partial" ? "Partially observed" : "Unavailable";
    details.append(metric("Restarts", unit.status === "ok" && unit.restarts != null ? numbers.format(unit.restarts) : "Unavailable"),
      metric("Kafka consumer lag", lag),
      metric(name === "listener" ? "Last registered run" : "Last fragment start", activityStamp(activity.last_input, activity)),
      metric(name === "listener" ? "Last Slurm acknowledgment" : "Last database commit", activityStamp(activity.last_success, activity)));
    card.append(heading, details);
    for (const [key, source, label, value] of [
      ["unit", unit, "Unit", state + " · " + (unit.restarts ?? "unknown") + " restarts"],
      ["kafka", kafka, "Consumer lag", lag],
      ["activity", activity, "Activity", "input " + activityStamp(activity.last_input, activity) +
        " · success " + activityStamp(activity.last_success, activity)]
    ]) {
      const id = name + "/" + key;
      if (source.status === "ok") {
        previousSources.set(id, {time: data.sampled_at, value});
      } else if (previousSources.has(id)) {
        const previous = previousSources.get(id);
        const stale = element("p", label + " · Last complete sample " + utc(previous.time) + " · ");
        const stamp = element("time", age(previous.time));
        stamp.dateTime = previous.time;
        stale.append(stamp, " · " + previous.value);
        stale.className = "stale-source";
        card.append(stale);
      }
    }
    const notes = [];
    if (unit.active_since) notes.push("Active since " + utc(unit.active_since));
    if (unit.result && unit.result !== "success") notes.push("Last unit result · " + unit.result);
    if (unit.error) notes.push(unit.error);
    if (activity.error) notes.push(activity.error);
    if (kafka.error) notes.push(kafka.error);
    for (const topic of kafka.topics || []) {
      const value = topic.lag == null ? "unknown lag" : numbers.format(topic.lag) + " messages";
      const partitions = topic.partitions == null ? "partition count unavailable" :
        topic.known_partitions + "/" + topic.partitions + " partitions observed";
      notes.push(topic.topic + " · " + value + " · " + partitions);
    }
    if (notes.length) {
      const more = document.createElement("details");
      more.dataset.service = name;
      more.open = Boolean(el("health-services").querySelector('details[data-service="' + name + '"][open]'));
      more.append(element("summary", "Source details"));
      for (const note of notes) more.append(element("p", note));
      card.append(more);
    }
    cards.push(card);
  }
  el("health-services").replaceChildren(...cards);
}

function showHealthStatus() {
  const status = el("health-status");
  status.classList.toggle("error", Boolean(healthError));
  const stamp = healthSample ? "Observed " + utc(healthSample.sampled_at) + " · " + age(healthSample.sampled_at) : "";
  const text = healthError ? "Health refresh failed. " + healthError + (stamp ? " · Last successful check: " + stamp : "") :
    stamp ? stamp + " · Both services, independent of the statistics filters." : "Checking services…";
  if (text !== status.textContent) status.textContent = text;
  for (const time of el("health-services").querySelectorAll(".stale-source time")) {
    const elapsed = age(time.dateTime);
    if (time.textContent !== elapsed) time.textContent = elapsed;
  }
}

async function refreshHealth() {
  if (healthRequest) return;
  healthRequest = true;
  try {
    const response = await fetch("/api/health", {cache: "no-store"});
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    drawHealth(data);
    healthSample = data;
    healthError = "";
  } catch (error) {
    healthError = error.message;
  } finally {
    healthRequest = false;
    showHealthStatus();
  }
}
