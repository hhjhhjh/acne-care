from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def render_dashboard_html(
    document: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("valid", "test"),
) -> str:
    items = select_review_items(document, splits=splits)
    serialized_items = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    dataset_fingerprint = review_dataset_fingerprint(document, splits=splits)
    return TEMPLATE.replace("__REVIEW_ITEMS__", serialized_items).replace(
        "__DATASET_FINGERPRINT__",
        dataset_fingerprint,
    )


def select_review_items(
    document: dict[str, Any],
    *,
    splits: tuple[str, ...],
) -> list[dict[str, Any]]:
    items = []
    for source_item in document["items"]:
        if source_item["split"] not in splits:
            continue
        item = dict(source_item)
        item["asset_path"] = (
            Path("assets") / item["split"] / Path(item["image_path"]).name
        ).as_posix()
        items.append(item)
    return items


def review_dataset_fingerprint(
    document: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("valid", "test"),
) -> str:
    items = select_review_items(document, splits=splits)
    serialized_items = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    return sha256(serialized_items.encode("utf-8")).hexdigest()[:12]


TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>acne-care 라벨 검토</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#151c31; --line:#2a3555; --text:#eef2ff; --muted:#9da8c7; --accent:#65d7c4; --warn:#ffbe5c; --danger:#ff7185; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    button,input,select,textarea { font:inherit; }
    header { display:flex; align-items:center; gap:18px; padding:16px 20px; border-bottom:1px solid var(--line); background:#10172a; }
    h1 { margin:0; font-size:19px; }
    .progress { color:var(--muted); }
    .actions { margin-left:auto; display:flex; gap:8px; }
    button,.file-label { border:1px solid var(--line); border-radius:8px; padding:8px 11px; background:#1d2742; color:var(--text); cursor:pointer; }
    button:hover,.file-label:hover { border-color:var(--accent); }
    .layout { display:grid; grid-template-columns:330px minmax(0,1fr) 330px; height:calc(100vh - 66px); }
    aside,.review { overflow:auto; background:var(--panel); }
    aside { border-right:1px solid var(--line); }
    .review { border-left:1px solid var(--line); padding:18px; }
    .filters { display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:12px; border-bottom:1px solid var(--line); }
    .filters input { grid-column:1/-1; }
    input,select,textarea { width:100%; border:1px solid var(--line); border-radius:7px; padding:8px; background:#0e1528; color:var(--text); }
    #item-list { padding:8px; }
    .item { width:100%; text-align:left; margin-bottom:6px; padding:10px; background:#121a2e; }
    .item.active { border-color:var(--accent); background:#1b2a42; }
    .item-line { display:flex; justify-content:space-between; gap:8px; }
    .filename { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .badge { flex-shrink:0; white-space:nowrap; padding:2px 6px; border-radius:999px; background:#283452; color:var(--muted); font-size:11px; }
    .badge.done { background:#17483f; color:#9af0d9; }
    .hints { margin-top:5px; color:var(--warn); font-size:12px; }
    main { min-width:0; display:flex; flex-direction:column; background:#080c17; }
    .viewer-head { padding:12px 16px; display:flex; gap:10px; align-items:center; border-bottom:1px solid var(--line); }
    #path { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .viewer { flex:1; min-height:0; display:grid; place-items:center; overflow:auto; padding:16px; }
    canvas { max-width:100%; max-height:100%; object-fit:contain; box-shadow:0 12px 40px #0009; }
    .nav { display:flex; justify-content:center; gap:8px; padding:10px; border-top:1px solid var(--line); }
    h2 { margin:0 0 12px; font-size:17px; }
    .meta { display:grid; grid-template-columns:auto 1fr; gap:7px 10px; margin-bottom:16px; }
    .meta dt { color:var(--muted); }
    .meta dd { margin:0; }
    .status-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .status-grid button.selected { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent) inset; }
    .status-grid button[data-status="needs_correction"] { color:#ffd08a; }
    .status-grid button[data-status="exclude"] { color:#ff9caa; }
    label.block { display:block; margin:16px 0 6px; color:var(--muted); }
    textarea { min-height:120px; resize:vertical; }
    .detail-list { padding-left:18px; color:var(--warn); }
    .shortcut { margin-top:16px; color:var(--muted); font-size:12px; }
    #import-file { display:none; }
    @media (max-width:1000px) { .layout { grid-template-columns:260px minmax(0,1fr); } .review { position:fixed; right:0; top:66px; bottom:0; width:310px; box-shadow:-10px 0 30px #0008; } main { margin-right:310px; } }
  </style>
</head>
<body>
  <header>
    <h1>acne-care 라벨 검토</h1>
    <div class="progress" id="progress"></div>
    <div class="actions">
      <label class="file-label" for="import-file">검토 결과 불러오기</label>
      <input id="import-file" type="file" accept="application/json">
      <button id="export">검토 결과 내보내기</button>
    </div>
  </header>
  <div class="layout">
    <aside>
      <div class="filters">
        <input id="search" placeholder="파일 이름 검색">
        <select id="split-filter"><option value="all">전체 분할</option><option value="valid">Validation</option><option value="test">Test</option></select>
        <select id="status-filter"><option value="all">전체 상태</option><option value="unreviewed">미검토</option><option value="ok">정상</option><option value="needs_correction">수정 필요</option><option value="uncertain">판단 보류</option><option value="exclude">제외 후보</option></select>
        <select id="hint-filter"><option value="all">모든 자동 검사</option><option value="flagged">검토 후보만</option><option value="clean">자동 경고 없음</option></select>
      </div>
      <div id="item-list"></div>
    </aside>
    <main>
      <div class="viewer-head"><strong id="counter"></strong><span id="path"></span></div>
      <div class="viewer"><canvas id="canvas"></canvas></div>
      <div class="nav"><button id="previous">← 이전</button><button id="toggle-boxes">상자 숨기기</button><button id="next">다음 →</button></div>
    </main>
    <section class="review">
      <h2>검토 기록</h2>
      <dl class="meta" id="meta"></dl>
      <div class="status-grid">
        <button data-status="ok">정상 <small>1</small></button>
        <button data-status="needs_correction">수정 필요 <small>2</small></button>
        <button data-status="uncertain">판단 보류 <small>3</small></button>
        <button data-status="exclude">제외 후보 <small>4</small></button>
        <button data-status="unreviewed">미검토로 변경</button>
      </div>
      <label class="block" for="note">수정 내용과 판단 근거</label>
      <textarea id="note" placeholder="예: 오른쪽 볼 병변 2개 라벨 누락"></textarea>
      <label class="block">자동 검사 상세</label>
      <ul class="detail-list" id="details"></ul>
      <p class="shortcut">방향키로 이동할 수 있습니다. 숫자 1–4로 상태를 지정합니다. 기록은 이 브라우저에 자동 저장됩니다.</p>
    </section>
  </div>
  <script>
    const items = __REVIEW_ITEMS__;
    const datasetFingerprint = "__DATASET_FINGERPRINT__";
    const storageKey = `acne-care-label-review-${datasetFingerprint}`;
    let reviews = JSON.parse(localStorage.getItem(storageKey) || "{}");
    let filtered = [];
    let current = 0;
    let showBoxes = true;
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    const controls = ["search", "split-filter", "status-filter", "hint-filter"];

    function statusOf(item) { return (reviews[item.image_path] || {}).status || "unreviewed"; }
    function applyFilters() {
      const query = document.getElementById("search").value.toLowerCase();
      const split = document.getElementById("split-filter").value;
      const status = document.getElementById("status-filter").value;
      const hint = document.getElementById("hint-filter").value;
      const selectedPath = filtered[current]?.image_path;
      filtered = items.filter(item => {
        if (split !== "all" && item.split !== split) return false;
        if (status !== "all" && statusOf(item) !== status) return false;
        if (hint === "flagged" && item.review_hints.length === 0) return false;
        if (hint === "clean" && item.review_hints.length !== 0) return false;
        return item.image_path.toLowerCase().includes(query);
      });
      current = Math.max(0, filtered.findIndex(item => item.image_path === selectedPath));
      render();
    }

    function render() {
      renderList();
      renderProgress();
      if (!filtered.length) {
        document.getElementById("path").textContent = "조건에 맞는 이미지가 없습니다.";
        context.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }
      const item = filtered[current];
      document.getElementById("counter").textContent = `${current + 1} / ${filtered.length}`;
      document.getElementById("path").textContent = item.image_path;
      const review = reviews[item.image_path] || {status:"unreviewed", note:""};
      document.getElementById("note").value = review.note || "";
      document.querySelectorAll("[data-status]").forEach(button => button.classList.toggle("selected", button.dataset.status === review.status));
      document.getElementById("meta").innerHTML = `<dt>분할</dt><dd>${item.split}</dd><dt>라벨 수</dt><dd>${item.annotation_count}</dd><dt>자동 검사</dt><dd>${item.review_hints.map(hintLabel).join(", ") || "없음"}</dd>`;
      document.getElementById("details").innerHTML = (item.details.length ? item.details : ["기계적으로 확인된 구조 오류가 없습니다."]).map(value => `<li>${escapeHtml(value)}</li>`).join("");
      drawItem(item);
    }

    function renderList() {
      document.getElementById("item-list").innerHTML = filtered.map((item, index) => {
        const status = statusOf(item);
        const done = status !== "unreviewed";
        return `<button class="item ${index === current ? "active" : ""}" data-index="${index}"><div class="item-line"><span class="filename">${escapeHtml(item.image_path.split("/").pop())}</span><span class="badge ${done ? "done" : ""}">${statusLabel(status)}</span></div><div class="hints">${item.annotation_count}개 · ${item.review_hints.map(hintLabel).join(", ") || "자동 경고 없음"}</div></button>`;
      }).join("");
      document.querySelectorAll(".item").forEach(button => button.addEventListener("click", () => { current = Number(button.dataset.index); render(); }));
    }

    function renderProgress() {
      const reviewed = items.filter(item => statusOf(item) !== "unreviewed").length;
      document.getElementById("progress").textContent = `검토 ${reviewed} / ${items.length}`;
    }

    function drawItem(item) {
      const image = new Image();
      image.onload = () => {
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        context.drawImage(image, 0, 0);
        if (!showBoxes) return;
        context.lineWidth = Math.max(2, image.naturalWidth / 320);
        context.font = `bold ${Math.max(12, image.naturalWidth / 42)}px sans-serif`;
        item.annotations.forEach(annotation => {
          const width = annotation.width * canvas.width;
          const height = annotation.height * canvas.height;
          const x = annotation.x_center * canvas.width - width / 2;
          const y = annotation.y_center * canvas.height - height / 2;
          context.strokeStyle = "#50f0d0";
          context.strokeRect(x, y, width, height);
          context.fillStyle = "#071510cc";
          context.fillRect(x, Math.max(0, y - 19), 32, 19);
          context.fillStyle = "#d8fff7";
          context.fillText(`#${annotation.line_number}`, x + 3, Math.max(14, y - 4));
        });
      };
      image.src = item.asset_path;
    }

    function move(offset) { if (!filtered.length) return; current = (current + offset + filtered.length) % filtered.length; render(); }
    function setStatus(status) {
      if (!filtered.length) return;
      const item = filtered[current];
      reviews[item.image_path] = {...(reviews[item.image_path] || {}), status, note:document.getElementById("note").value, updated_at:new Date().toISOString()};
      save(); render();
    }
    function save() { localStorage.setItem(storageKey, JSON.stringify(reviews)); }
    function statusLabel(status) { return ({unreviewed:"미검토",ok:"정상",needs_correction:"수정 필요",uncertain:"판단 보류",exclude:"제외 후보"})[status]; }
    function hintLabel(hint) { return ({missing_label:"라벨 파일 누락",invalid_label:"잘못된 라벨",unexpected_class:"예상 밖 클래스",empty_label:"빈 라벨",duplicate_box:"중복 상자",very_small_box:"매우 작은 상자",very_large_box:"매우 큰 상자",high_annotation_count:"라벨 수 이상치",unreadable_image:"이미지 읽기 실패"})[hint] || hint; }
    function escapeHtml(value) { const node=document.createElement("span"); node.textContent=value; return node.innerHTML; }

    controls.forEach(id => document.getElementById(id).addEventListener(id === "search" ? "input" : "change", applyFilters));
    document.getElementById("previous").addEventListener("click", () => move(-1));
    document.getElementById("next").addEventListener("click", () => move(1));
    document.getElementById("toggle-boxes").addEventListener("click", event => { showBoxes=!showBoxes; event.target.textContent=showBoxes ? "상자 숨기기" : "상자 보이기"; render(); });
    document.querySelectorAll("[data-status]").forEach(button => button.addEventListener("click", () => setStatus(button.dataset.status)));
    document.getElementById("note").addEventListener("input", event => { if (!filtered.length) return; const item=filtered[current]; reviews[item.image_path]={...(reviews[item.image_path] || {status:"unreviewed"}),note:event.target.value,updated_at:new Date().toISOString()}; save(); });
    document.getElementById("export").addEventListener("click", () => { const payload={schema_version:1,dataset_fingerprint:datasetFingerprint,exported_at:new Date().toISOString(),reviews}; const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}); const link=document.createElement("a"); link.href=URL.createObjectURL(blob); link.download="acne-care-label-review.json"; link.click(); URL.revokeObjectURL(link.href); });
    document.getElementById("import-file").addEventListener("change", async event => { const payload=JSON.parse(await event.target.files[0].text()); if (!payload.reviews) throw new Error("reviews field is missing"); if (payload.dataset_fingerprint !== datasetFingerprint) throw new Error("review file belongs to a different dataset version"); reviews=payload.reviews; save(); render(); });
    document.addEventListener("keydown", event => { if (event.target.matches("input,textarea,select")) return; if (event.key === "ArrowLeft") move(-1); if (event.key === "ArrowRight") move(1); const shortcuts={"1":"ok","2":"needs_correction","3":"uncertain","4":"exclude"}; if (shortcuts[event.key]) setStatus(shortcuts[event.key]); });
    applyFilters();
  </script>
</body>
</html>
"""
