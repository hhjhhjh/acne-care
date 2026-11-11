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

function reportBar(label, record, maxCount) {
  const row = element("li", "", "report-bar-row");
  const caption = element("span", "", "report-bar-label");
  caption.append(element("strong", label), captureDate(record));
  const track = element("span", "", "report-bar-track");
  track.setAttribute("aria-hidden", "true");
  const fill = element("span", "", "report-bar-fill");
  fill.style.width = `${100 * record.count / maxCount}%`;
  track.append(fill);
  row.append(caption, track, element("strong", `${record.count}개`, "report-bar-value"));
  return row;
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

export function renderComparisonReport({ previous, current, comparison }, changedLabels) {
  const { count_delta: delta, days_between: days, analysis_settings_match: sameSettings } = comparison;
  $("report-period").textContent = `${previous.captured_on} → ${current.captured_on} · ${formatInterval(days)}`;
  $("report-delta").textContent = formatCountDelta(delta);
  $("report-counts").textContent = `이전 ${previous.count}개 → 현재 ${current.count}개`;
  if (!sameSettings) {
    $("report-explanation").textContent = "분석 조건이 다르면 같은 사진에서도 탐지 개수가 달라질 수 있어요. 이 차이는 참고용으로 봐 주세요.";
  } else if (delta === 0) {
    $("report-explanation").textContent = "두 사진에서 탐지 개수가 같아요. 탐지 위치는 아래 사진에서 확인할 수 있어요.";
  } else {
    $("report-explanation").textContent = `현재 사진의 탐지 개수가 이전 사진보다 ${Math.abs(delta)}개 ${delta > 0 ? "많아요" : "적어요"}.`;
  }

  const maxCount = Math.max(1, previous.count, current.count);
  $("report-scale-max").textContent = `${maxCount}개`;
  $("report-bars").replaceChildren(reportBar("이전 기록", previous, maxCount), reportBar("현재 기록", current, maxCount));
  $("report-photos").replaceChildren(reportPhoto("이전 기록", previous), reportPhoto("현재 기록", current));

  const settingsStatus = $("report-settings-status");
  settingsStatus.classList.toggle("different", !sameSettings);
  settingsStatus.textContent = sameSettings
    ? "두 기록에 같은 모델과 탐지 설정을 사용했어요."
    : `두 기록의 ${changedLabels.join(", ")} 설정이 달라요.`;
  const fields = [
    ["모델", previous.model_id, current.model_id],
    ["탐지 기준 점수", previous.confidence_threshold, current.confidence_threshold],
    ["입력 크기", `${previous.image_size} × ${previous.image_size}`, `${current.image_size} × ${current.image_size}`],
  ];
  $("report-settings").replaceChildren(...fields.map(([label, before, after]) => {
    const row = element("div");
    row.append(element("dt", label), element("dd", before === after ? String(before) : `이전 ${before} / 현재 ${after}`));
    return row;
  }));
}
