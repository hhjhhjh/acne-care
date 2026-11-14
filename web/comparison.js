import { comparisonRecordLabel, formatCountDelta, formatInterval, renderComparisonChart, renderComparisonReport } from "./report.js";

const $ = id => document.getElementById(id);
const settingLabels = [
  ["model_id", "모델"], ["weights_sha256", "가중치 SHA-256"],
  ["confidence_threshold", "탐지 기준 점수"], ["image_size", "입력 크기"],
  ["nms_iou", "중복 상자 제거 기준"], ["device", "실행 장치"],
];

function changedSettingLabels(comparison) {
  return comparison.setting_differences.map(item => settingLabels.find(([field]) => field === item.field)?.[1] || item.field);
}

// Dependencies are passed explicitly; this module never uploads or changes a record.
export function createComparison({ api, showPage, openRecord }) {
  const records = new Map();
  const extraIds = new Set();
  let offset = 0, total = 0, loading = false, comparing = false;
  let result = null, limit = 2;

  function setURL() {
    if ($("comparison-page").hidden) return;
    const query = new URLSearchParams({ view: "comparison" });
    if (result?.records.length === 2) {
      query.set("previous", result.records[0].id);
      query.set("current", result.records[1].id);
    } else if (result) {
      for (const record of result.records) query.append("record", record.id);
    }
    if (!$("comparison-report").hidden) query.set("view", "report");
    history.replaceState(null, "", `/?${query}`);
  }

  function setReportView(show) {
    $("comparison-entry").hidden = show;
    $("comparison-report").hidden = !show;
  }

  function openReport() {
    if (!result || $("comparison-page").hidden) return;
    renderComparisonReport(result, changedSettingLabels);
    setReportView(true);
    setURL();
    $("report-title").focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function selectionError() {
    const previous = records.get($("previous-record").value);
    const current = records.get($("current-record").value);
    if (!previous || !current) return "비교할 기록 두 개를 선택해 주세요.";
    if (previous.id === current.id) return "서로 다른 기록 두 개를 선택해 주세요.";
    if (previous.captured_on > current.captured_on) return "현재 기록의 촬영일은 이전 기록과 같거나 이후여야 합니다.";
    return "";
  }

  function updateControls() {
    const working = loading || comparing;
    for (const id of ["previous-record", "current-record", "refresh-comparison", "more-comparison-records"]) $(id).disabled = working;
    $("compare-button").disabled = working || Boolean(selectionError());
    $("compare-button").textContent = comparing ? "비교 결과를 불러오는 중…" : "비교 결과 보기 →";
    $("more-comparison-records").hidden = offset >= total;
    const limitReached = extraIds.size + 2 >= limit;
    for (const checkbox of $("extra-comparison-records").querySelectorAll("input")) {
      checkbox.disabled = working || (limitReached && !checkbox.checked);
    }
  }

  function pickerStatus() {
    $("comparison-picker-status").textContent = total < 2
      ? "비교하려면 서로 다른 분석 기록이 두 개 이상 필요합니다. 사진 분석 후 다시 열어 주세요."
      : selectionError() || `선택한 기록 ${2 + extraIds.size}개 · 최대 ${limit}개까지 비교할 수 있어요.`;
  }

  function clearResult() {
    result = null;
    setReportView(false);
    $("comparison-result").hidden = true;
    setURL();
  }

  function populate(previousId, currentId) {
    const sorted = [...records.values()].sort((a, b) => b.captured_on.localeCompare(a.captured_on)
      || b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id));
    for (const id of ["previous-record", "current-record"]) {
      const placeholder = document.createElement("option");
      placeholder.value = ""; placeholder.textContent = "기록을 선택해 주세요.";
      $(id).replaceChildren(placeholder);
      for (const record of sorted) {
        const option = document.createElement("option");
        option.value = record.id;
        option.textContent = recordDescription(record);
        $(id).append(option);
      }
    }
    const current = records.get(currentId) || sorted[0];
    const previous = records.get(previousId) || sorted.find(record => record.id !== current?.id && record.captured_on <= current.captured_on);
    $("current-record").value = current?.id || "";
    $("previous-record").value = previous?.id || "";
    populateExtras();
  }

  function recordDescription(record) {
    return `${record.captured_on} · ${record.count}개 · 분석 ${new Date(record.created_at).toLocaleString("ko-KR")} · ${record.id.slice(0, 8)}`;
  }

  function populateExtras() {
    extraIds.delete($("previous-record").value);
    extraIds.delete($("current-record").value);
    const available = [...$("previous-record").options].slice(1).filter(option =>
      option.value !== $("previous-record").value && option.value !== $("current-record").value);
    $("extra-comparison-picker").hidden = !available.length || limit < 3;
    $("extra-comparison-records").replaceChildren(...available.map(option => {
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = option.value;
      checkbox.checked = extraIds.has(option.value);
      label.append(checkbox, document.createTextNode(option.textContent));
      return label;
    }));
  }

  async function loadRecords(reset, preferredCurrent = null) {
    if (loading || comparing) return;
    const previousId = $("previous-record").value;
    const currentId = preferredCurrent?.id || $("current-record").value;
    if (reset) { records.clear(); extraIds.clear(); offset = 0; total = 0; clearResult(); }
    if (preferredCurrent) records.set(preferredCurrent.id, preferredCurrent);
    loading = true;
    updateControls();
    $("comparison-picker-status").textContent = "저장된 기록을 불러오는 중입니다.";
    try {
      const listing = await api(`/api/analyses?limit=100&offset=${offset}`);
      listing.items.forEach(record => records.set(record.id, record));
      offset += listing.items.length;
      total = listing.total;
      populate(previousId, currentId);
      pickerStatus();
    } catch (error) {
      // Rebuild the choices even on failure so old options cannot select cleared records.
      populate(previousId, currentId);
      $("comparison-picker-status").textContent = error.message;
    } finally {
      loading = false;
      updateControls();
    }
  }

  function setImages(showAnnotated) {
    for (const [index, image] of Array.from($("comparison-images").querySelectorAll("img")).entries()) {
      const record = result.records[index];
      image.src = showAnnotated ? record.annotated_image_url : record.original_image_url;
      image.alt = `${comparisonRecordLabel(index, result.records.length)}의 ${showAnnotated ? "탐지 결과" : "원본 사진"}`;
    }
    for (const [id, selected] of [["comparison-annotated", showAnnotated], ["comparison-original", !showAnnotated]]) {
      $(id).classList.toggle("selected", selected);
      $(id).setAttribute("aria-pressed", String(selected));
    }
  }

  function render(comparisonResult, scroll) {
    result = comparisonResult;
    const { records: selected, steps } = result;
    const first = selected[0], last = selected.at(-1);
    selected.forEach(record => records.set(record.id, record));
    extraIds.clear();
    selected.slice(1, -1).forEach(record => extraIds.add(record.id));
    populate(first.id, last.id);
    if (extraIds.size) $("extra-comparison-picker").open = true;
    $("comparison-delta").textContent = formatCountDelta(last.count - first.count);
    $("comparison-counts").textContent = `첫 기록 ${first.count}개 → 마지막 기록 ${last.count}개`;
    $("comparison-interval").textContent = formatInterval(steps.reduce((days, step) => days + step.days_between, 0));
    const chart = $("comparison-trend");
    chart.hidden = selected.length <= 2;
    renderComparisonChart(chart, result, changedSettingLabels, openRecord);
    const settingsNotice = $("comparison-settings-notice");
    const changedLabels = [...new Set(steps.flatMap(changedSettingLabels))];
    settingsNotice.classList.toggle("different", changedLabels.length > 0);
    settingsNotice.textContent = changedLabels.length === 0
      ? "모델과 탐지 설정이 같습니다. 촬영 조건은 별도로 확인해 주세요."
      : `기록 사이에 ${changedLabels.join(", ")} 설정이 바뀌었어요. 탐지 개수 차이는 참고용으로 봐 주세요.`;
    $("comparison-images").replaceChildren(...selected.map((record, index) => comparisonPhoto(record, index, selected.length)));
    $("comparison-setting-headings").replaceChildren(...["분석 설정", ...selected.map((_, index) =>
      comparisonRecordLabel(index, selected.length))].map(label => {
      const heading = document.createElement("th");
      heading.scope = "col";
      heading.textContent = label;
      return heading;
    }));
    $("comparison-settings").replaceChildren();
    for (const [field, label] of settingLabels) {
      const row = document.createElement("tr"), heading = document.createElement("th");
      heading.scope = "row"; heading.textContent = label; row.append(heading);
      for (const record of selected) {
        const cell = document.createElement("td"); cell.textContent = String(record[field]); row.append(cell);
      }
      row.classList.toggle("different", selected.some(record => record[field] !== first[field]));
      $("comparison-settings").append(row);
    }
    setImages(true);
    $("comparison-result").hidden = false;
    setURL();
    if (scroll && !$("comparison-page").hidden) $("comparison-result").scrollIntoView({ block: "start" });
  }

  function comparisonPhoto(record, index, total) {
    const article = document.createElement("article");
    const heading = document.createElement("h3");
    heading.textContent = comparisonRecordLabel(index, total);
    const date = document.createElement("p");
    date.className = "muted";
    date.textContent = `촬영 ${recordDescription(record)}`;
    const frame = document.createElement("div");
    frame.className = "comparison-image-wrap";
    const image = document.createElement("img");
    image.addEventListener("error", () => {
      $("comparison-picker-status").textContent = "비교 사진을 불러오지 못했습니다. 저장된 사진 파일과 서버 연결을 확인해 주세요.";
    });
    frame.append(image);
    const details = document.createElement("button");
    details.type = "button";
    details.className = "text-button";
    details.textContent = "기록 상세 보기 ↗";
    details.setAttribute("aria-label", `${heading.textContent} 상세 보기`);
    details.addEventListener("click", () => openRecord(record.id));
    article.append(heading, date, frame, details);
    return article;
  }

  async function compare(ids, scroll = true) {
    if (loading || comparing) return;
    clearResult();
    comparing = true; updateControls();
    $("comparison-picker-status").textContent = "저장된 분석 결과를 비교하고 있습니다.";
    try {
      if (ids.length === 2) {
        const params = new URLSearchParams({ previous_id: ids[0] || "", current_id: ids[1] || "" });
        const { previous, current, comparison } = await api(`/api/comparisons?${params}`);
        render({ records: [previous, current], steps: [comparison] }, scroll);
      } else {
        const params = new URLSearchParams(ids.map(id => ["record_id", id]));
        render(await api(`/api/comparisons/series?${params}`), scroll);
      }
      pickerStatus();
    } catch (error) {
      $("comparison-picker-status").textContent = error.message;
    } finally {
      comparing = false; updateControls();
    }
  }

  $("comparison-form").addEventListener("submit", event => {
    event.preventDefault();
    const error = selectionError();
    if (error) { $("comparison-picker-status").textContent = error; return; }
    compare([$("previous-record").value, ...extraIds, $("current-record").value]);
  });
  for (const id of ["previous-record", "current-record"]) $(id).addEventListener("change", () => {
    clearResult(); populateExtras(); pickerStatus(); updateControls();
  });
  $("extra-comparison-records").addEventListener("change", event => {
    const checkbox = event.target;
    if (checkbox.checked) extraIds.add(checkbox.value);
    else extraIds.delete(checkbox.value);
    clearResult(); pickerStatus(); updateControls();
  });
  $("refresh-comparison").addEventListener("click", () => loadRecords(true));
  $("more-comparison-records").addEventListener("click", () => loadRecords(false));
  $("comparison-annotated").addEventListener("click", () => setImages(true));
  $("comparison-original").addEventListener("click", () => setImages(false));
  $("open-report").addEventListener("click", openReport);
  $("close-report").addEventListener("click", () => {
    setReportView(false);
    setURL();
    $("open-report").focus({ preventScroll: true });
    $("comparison-result").scrollIntoView({ block: "start" });
  });
  $("print-report").addEventListener("click", async () => {
    const images = [...$("report-photos").querySelectorAll("img")];
    await Promise.allSettled(images.map(image => image.decode()));
    if (!$("comparison-page").hidden && !$("comparison-report").hidden) window.print();
  });
  return {
    setLimit(value) { limit = value ?? 2; },
    async open(currentRecord = null) {
      showPage("comparison");
      await loadRecords(true, currentRecord);
      setURL();
    },
    async restore(ids, reportRequested = false) {
      showPage("comparison");
      await loadRecords(true);
      await compare(ids, false);
      if (reportRequested) openReport();
    },
  };
}
