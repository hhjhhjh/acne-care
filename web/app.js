"use strict";
import { createComparison } from "./comparison.js";

const $ = id => document.getElementById(id);
let selectedFile = null, previewURL = null, busy = false, ready = false, currentRecord = null;
let historyOffset = 0, historyTotal = 0, historyLoading = false;
const localDay = new Date();
const today = `${localDay.getFullYear()}-${String(localDay.getMonth()+1).padStart(2,"0")}-${String(localDay.getDate()).padStart(2,"0")}`;
$("captured-on").value = today;
$("captured-on").max = today;

function notice(message) { $("notice").textContent = message || ""; $("notice").hidden = !message; }
function showUploadError(message) {
  const error = $("upload-error");
  error.textContent = message;
  error.hidden = !message;
  if (message && !$("analysis-page").hidden && !$("analysis-entry").hidden) {
    error.scrollIntoView({ block: "nearest" });
  }
}
function updateButton() { $("analyze-button").disabled = !selectedFile || busy || !ready; }
async function api(path, options) {
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요."); }
  let payload;
  try { payload = await response.json(); } catch { throw new Error("서버 응답을 읽지 못했습니다. 다시 시도해 주세요."); }
  if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "입력 정보를 확인해 주세요.");
  return payload;
}
function clearPhoto() {
  selectedFile=null;
  if(previewURL) URL.revokeObjectURL(previewURL);
  previewURL=null;
  $("preview").hidden=true;$("preview").removeAttribute("src");
  $("upload-prompt").hidden=false;$("change-photo").hidden=true;
  $("photo-file").value="";$("file-name").textContent="선택한 사진이 없습니다.";$("file-size").textContent="";
  updateButton();
}
function rejectFile(message) {
  clearPhoto();
  showUploadError(message);
}
function selectFile(file) {
  if (busy || !file) return;
  showUploadError("");
  if (file.size > 10 * 1024 * 1024) { rejectFile("사진은 10MB 이하로 선택해 주세요."); return; }
  if (!/\.(jpe?g|png|webp)$/i.test(file.name)) { rejectFile("JPG, PNG, WEBP 사진을 선택해 주세요."); return; }
  if (previewURL) URL.revokeObjectURL(previewURL);
  selectedFile = file;
  previewURL = URL.createObjectURL(file);
  $("preview").src = previewURL;
  $("preview").hidden = false;
  $("upload-prompt").hidden = true;
  $("change-photo").hidden = false;
  $("file-name").textContent = file.name;
  $("file-size").textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  updateButton();
}
$("photo-file").addEventListener("change", event => selectFile(event.target.files[0]));
$("preview").addEventListener("error", () => { if(selectedFile) rejectFile("사진 미리보기를 읽지 못했습니다. 다른 사진을 선택해 주세요."); });
$("drop-zone").addEventListener("dragover", event => { event.preventDefault(); if (!busy) $("drop-zone").classList.add("dragover"); });
$("drop-zone").addEventListener("dragleave", () => $("drop-zone").classList.remove("dragover"));
$("drop-zone").addEventListener("drop", event => { event.preventDefault(); $("drop-zone").classList.remove("dragover"); selectFile(event.dataTransfer.files[0]); });

function showPage(name) {
  const changed = $(`${name}-page`).hidden;
  $("analysis-page").hidden = name !== "analysis";
  $("history-page").hidden = name !== "history";
  $("comparison-page").hidden = name !== "comparison";
  for (const page of ["analysis", "history", "comparison"]) {
    $(`${page}-tab`).classList.toggle("active", name === page);
    if (name === page) $(`${page}-tab`).setAttribute("aria-current", "page");
    else $(`${page}-tab`).removeAttribute("aria-current");
  }
  notice("");
  if (changed) window.scrollTo(0, 0);
}
function showAnalysisForm(focusUpload = false) {
  const leavingResult = !$("result").hidden;
  showPage("analysis");
  $("analysis-entry").hidden = false;
  $("result").hidden = true;
  currentRecord = null;
  history.replaceState(null, "", "/");
  if (leavingResult) {
    clearPhoto();
    $("captured-on").value = today;
    showUploadError("");
    window.scrollTo(0, 0);
  }
  if (focusUpload) {
    $("upload-title").focus({ preventScroll: true });
    $("upload-title").scrollIntoView({ block: "nearest" });
  }
}
const comparison = createComparison({ api, showPage, openRecord });
$("analysis-tab").addEventListener("click", () => showAnalysisForm());
$("history-tab").addEventListener("click", () => { showPage("history"); loadHistory(true); });
$("comparison-tab").addEventListener("click", () => comparison.open());
$("compare-current").addEventListener("click", () => comparison.open(currentRecord));
$("refresh-history").addEventListener("click", () => loadHistory(true));
$("load-more").addEventListener("click", () => loadHistory(false));
$("start-analysis").addEventListener("click", () => showAnalysisForm(true));
$("new-analysis").addEventListener("click", () => showAnalysisForm(true));

function setResultImage(annotated) {
  if (!currentRecord) return;
  $("result-image").src = annotated ? currentRecord.annotated_image_url : currentRecord.original_image_url;
  $("result-image").alt = annotated ? "여드름 탐지 결과" : "분석에 사용한 원본 사진";
  $("show-annotated").classList.toggle("selected",annotated);
  $("show-original").classList.toggle("selected",!annotated);
  $("show-annotated").setAttribute("aria-pressed",String(annotated));
  $("show-original").setAttribute("aria-pressed",String(!annotated));
}
$("show-annotated").addEventListener("click", () => setResultImage(true));
$("show-original").addEventListener("click", () => setResultImage(false));
$("result-image").addEventListener("error", () => notice("저장된 결과 이미지를 불러오지 못했습니다. 서버와 저장 폴더를 확인해 주세요."));

function renderResult(record, scroll=true) {
  currentRecord = record;
  showPage("analysis");
  $("analysis-entry").hidden = true;
  $("result").hidden = false;
  $("result-date").textContent = `촬영일 ${record.captured_on.replaceAll("-", ".")}`;
  $("result-count").textContent = record.count;
  $("result-message").textContent = record.count ? "상자는 모델이 여드름으로 탐지한 위치입니다. 누락되거나 잘못 탐지된 부분이 있을 수 있어요." : "모델이 탐지한 여드름이 없습니다. 피부 상태가 정상이라는 판정을 뜻하지는 않습니다.";
  $("result-detection-details").hidden = !record.count;
  $("result-detection-details").open = false;
  $("result-model-details").open = false;
  $("detection-list").replaceChildren();
  record.detections.forEach((detection,index) => {
    const row=document.createElement("li"), label=document.createElement("span"), value=document.createElement("span");
    label.textContent=`탐지 ${index+1}`;value.textContent=`${(detection.confidence*100).toFixed(1)}%`;
    row.append(label,value);$("detection-list").append(row);
  });
  $("model-details").replaceChildren();
  for (const [name,value] of [["분석 시각",new Date(record.created_at).toLocaleString("ko-KR")],["모델",record.model_id],["탐지 기준 점수",record.confidence_threshold],["입력 크기",`${record.image_size} × ${record.image_size}`],["실행 장치",record.device],["분석 이미지",`${record.width} × ${record.height}`],["가중치 SHA-256",record.weights_sha256]]) {
    const term=document.createElement("dt"), description=document.createElement("dd");term.textContent=name;description.textContent=value;$("model-details").append(term,description);
  }
  setResultImage(true);
  history.replaceState(null,"",`/?analysis=${encodeURIComponent(record.id)}`);
  if (scroll) {
    $("result-title").focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }
}
async function openRecord(id, scroll=true) {
  try { renderResult(await api(`/api/analyses/${encodeURIComponent(id)}`), scroll); }
  catch(error) { notice(error.message || "기록을 불러오지 못했습니다."); }
}

$("upload-form").addEventListener("submit",async event => {
  event.preventDefault();if(!selectedFile || busy || !ready) return;
  busy=true;updateButton();showUploadError("");
  $("photo-file").disabled=true;$("captured-on").disabled=true;
  $("analysis-progress").hidden=false;$("analyze-label").textContent="분석하고 있어요…";
  const form = new FormData();
  form.append("file", selectedFile);
  form.append("captured_on", $("captured-on").value);
  try { renderResult(await api("/api/analyses",{method:"POST",body:form})); }
  catch(error) { showUploadError(error.message || "서버에 연결할 수 없습니다. 실행 상태를 확인해 주세요."); }
  finally {
    busy = false;
    $("photo-file").disabled = false;
    $("captured-on").disabled = false;
    $("analysis-progress").hidden = true;
    $("analyze-label").textContent = "사진 분석하고 기록하기";
    updateButton();
  }
});

async function loadHistory(reset) {
  if(historyLoading) return;historyLoading=true;$("load-more").disabled=true;
  if(reset) {historyOffset=0;$("history-grid").replaceChildren();$("history-count").textContent="기록을 불러오는 중입니다.";$("history-empty").hidden=true;$("load-more").hidden=true;}
  try {
    const listing=await api(`/api/analyses?limit=12&offset=${historyOffset}`);
    historyTotal=listing.total;
    for(const record of listing.items) {
      const card=document.createElement("button");card.className="history-card";
      const image=document.createElement("img");image.src=record.original_image_url;image.alt=`${record.captured_on} 촬영 사진`;image.loading="lazy";
      const info=document.createElement("div");info.className="history-info";
      const title=document.createElement("strong");title.textContent=record.captured_on.replaceAll("-", ".");
      const detail=document.createElement("p");detail.textContent=`탐지 ${record.count}개`;
      const action=document.createElement("span");action.textContent="결과 보기 ↗";detail.append(action);info.append(title,detail);card.append(image,info);
      card.addEventListener("click",()=>openRecord(record.id));$("history-grid").append(card);
    }
    historyOffset+=listing.items.length;
    $("history-count").textContent=`총 ${historyTotal}개의 분석 기록`;
    $("history-empty").hidden=historyTotal!==0;
    $("load-more").hidden=historyOffset>=historyTotal;
  } catch(error) {notice(error.message || "기록을 불러오지 못했습니다.");$("history-count").textContent="기록을 불러오지 못했습니다.";}
  finally {historyLoading=false;$("load-more").disabled=false;}
}

async function start() {
  try {
    await api("/api/health");
    ready = true;
    $("model-status").textContent = "AI 모델 준비 완료";
  }
  catch {$("model-status").textContent="서버 연결 확인 필요";$("model-status").classList.add("error");notice("분석 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.");}
  updateButton();
  const params=new URLSearchParams(location.search);
  if(params.has("previous") || params.has("current")) {
    await comparison.restore(params.get("previous"), params.get("current"), params.get("view") === "report");
  } else if(params.get("view") === "comparison") {
    await comparison.open();
  } else if(params.get("analysis") && ready) {
    await openRecord(params.get("analysis"), false);
  }
}
start();
