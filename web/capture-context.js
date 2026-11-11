const fields = [
  ["capture_lighting", "조명"],
  ["capture_angle", "각도"],
  ["capture_distance", "거리"],
  ["capture_filter", "필터"],
  ["capture_note", "메모"],
];

export function configureCaptureContext({ capture_options, capture_note_limit }) {
  for (const [name, choices] of Object.entries(capture_options)) {
    const select = document.getElementById(name);
    for (const choice of choices) select.add(new Option(choice, choice));
  }
  document.getElementById("capture_note").maxLength = capture_note_limit;
}

export function renderCaptureContext(container, record) {
  const list = document.createElement("dl");
  list.className = "capture-context-list";
  for (const [name, label] of fields) {
    if (!record[name]) continue;
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = record[name];
    list.append(term, description);
  }
  if (list.childElementCount) {
    container.replaceChildren(list);
  } else {
    const empty = document.createElement("p");
    empty.className = "capture-context-empty";
    empty.textContent = "촬영 조건과 메모 미기록";
    container.replaceChildren(empty);
  }
}
