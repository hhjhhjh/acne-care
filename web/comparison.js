const $ = id => document.getElementById(id);
const settingLabels = [
  ["model_id", "모델"], ["weights_sha256", "가중치 SHA-256"],
  ["confidence_threshold", "탐지 기준 점수"], ["image_size", "입력 크기"],
  ["nms_iou", "중복 상자 제거 기준"], ["device", "실행 장치"],
];

// Dependencies are passed explicitly; this module never uploads or changes a record.
export function createComparison({ api, showPage, openRecord }) {
  const records = new Map();
  let offset = 0, total = 0, loading = false, comparing = false;
  let result = null, annotated = true;

  function setURL() {
    if ($("comparison-page").hidden) return;
    const query = result
      ? `?previous=${encodeURIComponent(result.previous.id)}&current=${encodeURIComponent(result.current.id)}`
      : "?view=comparison";
    history.replaceState(null, "", `/${query}`);
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
  }

  function pickerStatus() {
    $("comparison-picker-status").textContent = total < 2
      ? "비교하려면 서로 다른 분석 기록이 두 개 이상 필요합니다. 사진 분석 후 다시 열어 주세요."
      : selectionError() || `선택 가능한 기록 ${records.size}개 / 전체 ${total}개 · 같은 날짜의 기록은 분석 시각과 기록 ID로 구분합니다.`;
  }

  function clearResult() {
    result = null;
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
        option.textContent = `${record.captured_on} · ${record.count}개 · 분석 ${new Date(record.created_at).toLocaleString("ko-KR")} · ${record.id.slice(0, 8)}`;
        $(id).append(option);
      }
    }
    const current = records.get(currentId) || sorted[0];
    const previous = records.get(previousId) || sorted.find(record => record.id !== current?.id && record.captured_on <= current.captured_on);
    $("current-record").value = current?.id || "";
    $("previous-record").value = previous?.id || "";
  }

  async function loadRecords(reset, preferredCurrent = null) {
    if (loading || comparing) return;
    const previousId = $("previous-record").value;
    const currentId = preferredCurrent?.id || $("current-record").value;
    if (reset) { records.clear(); offset = 0; total = 0; clearResult(); }
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
    annotated = showAnnotated;
    if (!result) return;
    for (const side of ["previous", "current"]) {
      $(`${side}-image`).src = result[side][annotated ? "annotated_image_url" : "original_image_url"];
      $(`${side}-image`).alt = `${side === "previous" ? "이전" : "현재"} 기록의 ${annotated ? "탐지 결과" : "원본 사진"}`;
    }
    for (const [id, selected] of [["comparison-annotated", annotated], ["comparison-original", !annotated]]) {
      $(id).classList.toggle("selected", selected);
      $(id).setAttribute("aria-pressed", String(selected));
    }
  }

  function render(comparisonResult, scroll) {
    result = comparisonResult;
    const { previous, current, comparison } = result;
    records.set(previous.id, previous); records.set(current.id, current);
    populate(previous.id, current.id);
    const delta = comparison.count_delta;
    $("comparison-delta").textContent = delta === 0 ? "변화 없음" : `${delta > 0 ? "+" : "−"}${Math.abs(delta)}개`;
    $("comparison-counts").textContent = `이전 ${previous.count}개 → 현재 ${current.count}개`;
    $("comparison-interval").textContent = comparison.days_between === 0 ? "같은 촬영일의 기록" : `촬영일 간격 ${comparison.days_between}일`;
    const settingsNotice = $("comparison-settings-notice");
    settingsNotice.classList.toggle("different", !comparison.analysis_settings_match);
    const changedLabels = comparison.setting_differences.map(item => settingLabels.find(([field]) => field === item.field)?.[1] || item.field);
    settingsNotice.textContent = comparison.analysis_settings_match
      ? "모델과 탐지 설정이 같습니다. 촬영 조건은 별도로 확인해 주세요."
      : `분석 조건이 다릅니다: ${changedLabels.join(", ")}. 개수 차이를 같은 분석 조건의 변화로 해석하지 마세요.`;
    for (const [side, record] of [["previous", previous], ["current", current]]) {
      $(`${side}-date`).textContent = `촬영 ${record.captured_on} · 분석 ${new Date(record.created_at).toLocaleString("ko-KR")} · ${record.id.slice(0, 8)}`;
      $(`${side}-count`).textContent = `탐지 ${record.count}개`;
    }
    $("comparison-settings").replaceChildren();
    for (const [field, label] of settingLabels) {
      const row = document.createElement("tr"), heading = document.createElement("th");
      heading.scope = "row"; heading.textContent = label; row.append(heading);
      for (const record of [previous, current]) {
        const cell = document.createElement("td"); cell.textContent = String(record[field]); row.append(cell);
      }
      row.classList.toggle("different", previous[field] !== current[field]);
      $("comparison-settings").append(row);
    }
    setImages(true);
    $("comparison-result").hidden = false;
    setURL();
    if (scroll && !$("comparison-page").hidden) $("comparison-result").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function compare(previousId, currentId, scroll = true) {
    if (loading || comparing) return;
    clearResult();
    comparing = true; updateControls();
    $("comparison-picker-status").textContent = "저장된 분석 결과를 비교하고 있습니다.";
    try {
      const params = new URLSearchParams({ previous_id: previousId || "", current_id: currentId || "" });
      render(await api(`/api/comparisons?${params}`), scroll);
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
    compare($("previous-record").value, $("current-record").value);
  });
  for (const id of ["previous-record", "current-record"]) $(id).addEventListener("change", () => {
    clearResult(); pickerStatus(); updateControls();
  });
  $("refresh-comparison").addEventListener("click", () => loadRecords(true));
  $("more-comparison-records").addEventListener("click", () => loadRecords(false));
  $("comparison-annotated").addEventListener("click", () => setImages(true));
  $("comparison-original").addEventListener("click", () => setImages(false));
  for (const side of ["previous", "current"]) {
    $(`open-${side}`).addEventListener("click", () => { if (result) openRecord(result[side].id); });
    $(`${side}-image`).addEventListener("error", () => {
      if (result) $("comparison-picker-status").textContent = "비교 사진을 불러오지 못했습니다. 저장된 사진 파일과 서버 연결을 확인해 주세요.";
    });
  }

  return {
    async open(currentRecord = null) {
      showPage("comparison");
      await loadRecords(true, currentRecord);
      setURL();
    },
    async restore(previousId, currentId) {
      showPage("comparison");
      await loadRecords(true);
      await compare(previousId, currentId, false);
    },
  };
}
