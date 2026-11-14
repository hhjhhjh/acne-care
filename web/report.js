import { renderCaptureContext } from "./capture-context.js";

const $ = id => document.getElementById(id);

export function formatCountDelta(delta) {
  return delta === 0 ? "변화 없음" : `${delta > 0 ? "+" : "−"}${Math.abs(delta)}개`;
}

export function formatInterval(days) {
  return days === 0 ? "같은 촬영일의 기록" : `촬영일 간격 ${days}일`;
}

function element(tag, text = "", className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function captureDate(record) {
  const date = element("time", record.captured_on.replaceAll("-", "."));
  date.dateTime = record.captured_on;
  return date;
}

export function comparisonRecordLabel(index, total) {
  if (total === 2) return index === 0 ? "이전 기록" : "현재 기록";
  return `${index + 1}번째 기록`;
}

function reportBar(label, record, maxCount, changedLabels, openRecord) {
  const row = element("li", "", "report-bar-row");
  const caption = element("span", "", "report-bar-label");
  const title = openRecord ? element("button", label, "trend-record-button") : element("strong", label);
  if (openRecord) {
    title.type = "button";
    title.setAttribute("aria-label", `${label} 상세 보기 · ${record.captured_on} · 탐지 ${record.count}개`);
    title.addEventListener("click", () => openRecord(record.id));
  }
  caption.append(title, captureDate(record));
  if (changedLabels.length) {
    const change = element("span", "설정 변경", "comparison-step-change");
    change.title = changedLabels.join(", ");
    caption.append(change);
  }
  const track = element("span", "", "report-bar-track");
  track.setAttribute("aria-hidden", "true");
  const fill = element("span", "", "report-bar-fill");
  fill.style.width = `${100 * record.count / maxCount}%`;
  track.append(fill);
  row.append(caption, track, element("strong", `${record.count}개`, "report-bar-value"));
  return row;
}

export function renderComparisonChart(container, { records, steps }, getChangedLabels, openRecord) {
  const maxCount = Math.max(1, ...records.map(record => record.count));
  container.replaceChildren(...records.map((record, index) => reportBar(
    comparisonRecordLabel(index, records.length), record, maxCount,
    index === 0 ? [] : getChangedLabels(steps[index - 1]), openRecord,
  )));
  return maxCount;
}

function reportPhoto(label, record) {
  const figure = element("figure", "", "report-photo");
  const caption = element("figcaption");
  const heading = element("div", "", "report-photo-label");
  heading.append(element("strong", label), element("strong", `${record.count}개`));
  const date = element("p", "촬영 ", "report-photo-date");
  date.append(captureDate(record), document.createTextNode(` · 기록 ${record.id.slice(0, 8)}`));
  caption.append(heading, date);

  const frame = element("div", "", "report-photo-frame");
  const image = element("img");
  image.alt = `${label} 사진의 여드름 탐지 위치`;
  const fallback = element("p", "저장된 탐지 사진을 불러오지 못했어요.");
  fallback.hidden = true;
  image.addEventListener("error", () => {
    image.hidden = true;
    fallback.hidden = false;
  }, { once: true });
  image.src = record.annotated_image_url;
  frame.append(image, fallback);
  const context = element("div", "", "report-photo-context");
  renderCaptureContext(context, record);
  figure.append(caption, frame, context);
  return figure;
}

export function renderComparisonReport(result, getChangedLabels) {
  const { records, steps } = result;
  const first = records[0], last = records.at(-1);
  const delta = last.count - first.count;
  const days = steps.reduce((total, step) => total + step.days_between, 0);
  const changes = steps.flatMap((step, index) => step.analysis_settings_match
    ? [] : [`${index + 1} → ${index + 2}번째 기록 (${getChangedLabels(step).join(", ")})`]);
  const sameSettings = changes.length === 0;
  $("comparison-report").classList.toggle("multi-report", records.length > 2);
  $("report-period").textContent = `${first.captured_on} → ${last.captured_on} · ${formatInterval(days)}`;
  $("report-delta").textContent = formatCountDelta(delta);
  $("report-counts").textContent = `첫 기록 ${first.count}개 → 마지막 기록 ${last.count}개`;
  if (!sameSettings) {
    $("report-explanation").textContent = "분석 조건이 다르면 같은 사진에서도 탐지 개수가 달라질 수 있어요. 이 차이는 참고용으로 봐 주세요.";
  } else if (delta === 0) {
    $("report-explanation").textContent = "첫 기록과 마지막 기록의 탐지 개수가 같아요.";
  } else {
    $("report-explanation").textContent = `마지막 기록은 첫 기록보다 ${Math.abs(delta)}개 ${delta > 0 ? "많아요" : "적어요"}.`;
  }

  const maxCount = renderComparisonChart($("report-bars"), result, getChangedLabels);
  $("report-scale-max").textContent = `${maxCount}개`;
  $("report-photos").replaceChildren(...records.map((record, index) =>
    reportPhoto(comparisonRecordLabel(index, records.length), record)));

  const settingsStatus = $("report-settings-status");
  settingsStatus.classList.toggle("different", !sameSettings);
  settingsStatus.textContent = sameSettings
    ? "모든 기록에 같은 모델과 탐지 설정을 사용했어요."
    : `분석 설정이 달라진 구간은 ${changes.join(" · ")}예요.`;
  const fields = [
    ["모델", record => record.model_id],
    ["탐지 기준 점수", record => record.confidence_threshold],
    ["입력 크기", record => `${record.image_size} × ${record.image_size}`],
  ];
  $("report-settings").replaceChildren(...fields.map(([label, getValue]) => {
    const row = element("div");
    const values = records.map(getValue);
    const description = values.every(value => value === values[0]) ? String(values[0])
      : values.map((value, index) => `${comparisonRecordLabel(index, records.length)} ${value}`).join(" / ");
    row.append(element("dt", label), element("dd", description));
    return row;
  }));
}
