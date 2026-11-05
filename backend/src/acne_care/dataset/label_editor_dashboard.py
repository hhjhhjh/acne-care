from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from acne_care.dataset.label_review_dashboard import review_dataset_fingerprint


ACTIONABLE_STATUSES = {"needs_correction", "exclude"}


def render_label_editor_html(
    document: dict[str, Any],
    review_results: dict[str, Any] | None = None,
    *,
    include_all: bool = False,
) -> str:
    if include_all:
        review_results = {"reviews": {}, "dataset_fingerprint": review_dataset_fingerprint(document, splits=("train", "valid", "test"))}
    if review_results is None:
        raise ValueError("review results are required unless include_all is enabled")
    reviews = review_results.get("reviews")
    if not isinstance(reviews, dict):
        raise ValueError("review results must contain a reviews object")
    items = []
    for source_item in document["items"]:
        review = reviews.get(source_item["image_path"], {})
        if not include_all and review.get("status") not in ACTIONABLE_STATUSES:
            continue
        item = dict(source_item)
        item["asset_path"] = source_item.get("asset_path") or (
            Path("assets") / item["split"] / Path(item["image_path"]).name
        ).as_posix()
        item["review"] = review or {"status": "unreviewed", "note": ""}
        items.append(item)
    serialized = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    fingerprint = str(review_results.get("dataset_fingerprint", "unknown"))
    return TEMPLATE.replace("__EDITOR_ITEMS__", serialized).replace("__ALL_ITEMS__", "true" if include_all else "false").replace(
        "__DATASET_FINGERPRINT__",
        fingerprint,
    )


TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>acne-care 라벨 편집</title>
  <style>
    :root { color-scheme:dark; --bg:#080d19; --panel:#141c31; --line:#2a3658; --text:#eef2ff; --muted:#9ca9ca; --mint:#55e3c2; --yellow:#ffd166; --red:#ff7185; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    button,input,textarea,select { font:inherit; }
    button { border:1px solid var(--line); border-radius:8px; padding:8px 11px; color:var(--text); background:#1d2845; cursor:pointer; }
    button:hover,button.active { border-color:var(--mint); }
    header { height:66px; display:flex; align-items:center; gap:18px; padding:14px 20px; background:#10172a; border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:19px; }
    #progress { color:var(--muted); }
    .header-actions { margin-left:auto; display:flex; gap:8px; align-items:center; }
    .file-button { border:1px solid var(--line); border-radius:8px; padding:8px 11px; cursor:pointer; }
    #import-file { display:none; }
    #save-status { color:var(--muted); font-size:12px; }
    #save-status.error { color:var(--red); }
    .filters { display:grid; gap:7px; margin-bottom:12px; position:sticky; top:0; background:var(--panel); padding-bottom:8px; }
    .filters input,.filters select { min-width:0; width:100%; padding:8px; border:1px solid var(--line); border-radius:6px; color:var(--text); background:var(--bg); }
    #filter-count { color:var(--muted); font-size:12px; }
    .layout { display:grid; grid-template-columns:320px minmax(0,1fr) 330px; height:calc(100vh - 66px); }
    aside,.inspector { overflow:auto; background:var(--panel); }
    aside { border-right:1px solid var(--line); padding:10px; }
    .inspector { border-left:1px solid var(--line); padding:18px; }
    .item { width:100%; text-align:left; margin-bottom:7px; background:#11192d; }
    .item.active { border-color:var(--mint); background:#1a2942; }
    .item-line { display:flex; align-items:center; gap:8px; }
    .filename { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .badge { flex-shrink:0; padding:2px 7px; border-radius:999px; color:var(--muted); background:#2a3552; font-size:11px; }
    .badge.done { color:#9df3d8; background:#174a3e; }
    .badge.exclude { color:#ffc0ca; background:#542537; }
    .note-line { margin-top:5px; color:var(--yellow); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    main { min-width:0; display:flex; flex-direction:column; }
    .viewer-head { padding:11px 15px; border-bottom:1px solid var(--line); display:flex; gap:10px; }
    #path { color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .viewer { flex:1; min-height:0; overflow:auto; padding:14px; }
    canvas { display:block; margin:auto; box-shadow:0 12px 40px #000a; cursor:crosshair; touch-action:none; }
    .toolbar { display:flex; flex-wrap:wrap; justify-content:center; gap:7px; padding:10px; border-top:1px solid var(--line); }
    .toolbar .danger { color:#ffabb7; }
    h2 { margin:0 0 14px; font-size:17px; }
    .legend { display:flex; gap:14px; color:var(--muted); margin-bottom:15px; }
    .dot { display:inline-block; width:11px; height:11px; margin-right:5px; border:2px solid; }
    .original { border-color:var(--mint); }
    .added { border-color:var(--yellow); }
    dl { display:grid; grid-template-columns:auto 1fr; gap:7px 10px; }
    dt { color:var(--muted); } dd { margin:0; }
    textarea { width:100%; min-height:105px; resize:vertical; border:1px solid var(--line); border-radius:8px; padding:9px; background:#0d1427; color:var(--text); }
    label.block { display:block; margin:16px 0 6px; color:var(--muted); }
    .exclude-row { display:flex; align-items:center; gap:8px; margin-top:15px; color:#ffb6c1; }
    .exclude-row input { width:17px; height:17px; }
    #complete { width:100%; margin-top:15px; padding:11px; }
    #complete.done { background:#165243; border-color:var(--mint); }
    .help { margin-top:17px; color:var(--muted); font-size:12px; }
    @media (max-width:1100px) { .layout { grid-template-columns:210px minmax(0,1fr) 260px; } header { gap:10px; padding:10px; } h1 { font-size:16px; } .header-actions { gap:4px; } .header-actions button,.file-button { padding:7px; font-size:12px; } #save-status { max-width:140px; } }
    @media (max-width:780px) {
      header { height:auto; min-height:66px; flex-wrap:wrap; }
      .header-actions { width:100%; justify-content:flex-end; }
      .layout { display:flex; flex-direction:column; height:auto; }
      aside { overflow:visible; border-bottom:1px solid var(--line); }
      .filters { grid-template-columns:1fr 1fr; margin:0; padding:0; position:static; }
      #search,#filter-count { grid-column:1/-1; }
      #item-list { display:none; }
      main { min-height:520px; }
      .viewer { flex:none; height:55vh; min-height:300px; }
      .inspector { overflow:visible; border-top:1px solid var(--line); border-left:none; }
      .toolbar { position:sticky; bottom:0; z-index:2; background:var(--bg); }
      #path { font-size:11px; }
    }
  </style>
</head>
<body>
  <header><h1>acne-care 라벨 편집</h1><span id="progress"></span><div class="header-actions"><span id="save-status"></span><label class="file-button" for="import-file">편집 결과 불러오기</label><input id="import-file" type="file" accept="application/json"><button id="export">편집 결과 내보내기</button></div></header>
  <div class="layout">
    <aside><div class="filters"><input id="search" placeholder="파일 이름 검색"><select id="split-filter"><option value="all">전체 분할</option><option value="train">Train</option><option value="valid">Validation</option><option value="test">Test</option></select><select id="status-filter"><option value="all">전체 상태</option><option value="pending">미완료</option><option value="completed">완료</option><option value="excluded">제외 선택</option><option value="empty">현재 상자 없음</option></select><span id="filter-count"></span></div><div id="item-list"></div></aside>
    <main>
      <div class="viewer-head"><strong id="counter"></strong><span id="path"></span></div>
      <div class="viewer"><canvas id="canvas"></canvas></div>
      <div class="toolbar">
        <button id="previous">← 이전</button>
        <button id="select-mode">상자 선택</button>
        <button id="draw-mode" class="active">상자 추가</button>
        <button id="delete-box" class="danger">선택 상자 삭제</button>
        <button id="undo">되돌리기</button>
        <button id="toggle-boxes">상자 숨기기</button>
        <label>확대 <select id="zoom"><option value="1">화면 맞춤</option><option value="1.5">150%</option><option value="2">200%</option><option value="3">300%</option><option value="4">400%</option></select></label>
        <button id="reset" class="danger">현재 이미지 초기화</button>
        <button id="next">다음 →</button>
      </div>
    </main>
    <section class="inspector">
      <h2>편집 정보</h2>
      <div class="legend"><span><i class="dot original"></i>기존 상자</span><span><i class="dot added"></i>추가 상자</span></div>
      <dl id="meta"></dl>
      <label class="block" for="note">검토 메모</label>
      <textarea id="note"></textarea>
      <label class="exclude-row"><input id="exclude" type="checkbox"> 추가 학습 데이터에서 이미지 제외</label>
      <button id="complete">이 이미지 수정 완료</button>
      <button id="complete-next" style="width:100%;margin-top:8px">완료 후 다음 이미지</button>
      <p class="help">D: 상자 추가. S: 상자 선택. 선택 모드에서 상자를 드래그하면 이동하고 모서리 손잡이를 드래그하면 크기를 조절합니다. Delete: 삭제. ← →: 이미지 이동. 상자가 없거나 수정하지 않은 사진도 직접 완료를 표시합니다.</p>
      <p class="help" id="dataset-help">작업은 이 브라우저에 자동 저장됩니다. 종료 전 편집 결과를 JSON으로 내보내세요. 불러오기로 다른 브라우저에서도 이어서 작업할 수 있습니다. 원본 파일은 변경하지 않습니다.</p>
    </section>
  </div>
  <script>
    const items = __EDITOR_ITEMS__;
    const allItems = __ALL_ITEMS__;
    const datasetFingerprint = "__DATASET_FINGERPRINT__";
    const storageKey = `acne-care-label-editor-${datasetFingerprint}`;
    let saved = {};
    let startupError = "";
    try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}"); validateEdits(saved); }
    catch (error) { startupError=`저장 데이터 복원 실패: ${error.message}. 내보낸 JSON을 불러오세요.`; saved={}; }
    const edits = {};
    const history = {};
    let current = 0;
    let mode = "draw";
    let selected = null;
    let dragStart = null;
    let dragCurrent = null;
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");
    let currentImage = null;
    let loadedPath = null;
    let requestedPath = null;
    let gesture = null;
    let showBoxes = true;
    let filtered = items.map((_, index) => index);

    function validateEdits(value) {
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("edits 형식 오류");
      const known = new Set(items.map(item => item.image_path));
      for (const [path, edit] of Object.entries(value)) {
        if (!known.has(path)) throw new Error("다른 데이터의 이미지가 포함되어 있습니다");
        if (!edit || !Array.isArray(edit.annotations) || typeof edit.completed!=="boolean" || typeof edit.excluded!=="boolean" || typeof edit.note!=="string") throw new Error("편집 항목 형식 오류");
        for (const box of edit.annotations) {
          const values=[box.x_center,box.y_center,box.width,box.height];
          if(box.class_id!==0 || !values.every(value=>typeof value==="number" && Number.isFinite(value)) || box.width<=0 || box.height<=0 || box.x_center-box.width/2 < -1e-9 || box.y_center-box.height/2 < -1e-9 || box.x_center+box.width/2 > 1+1e-9 || box.y_center+box.height/2 > 1+1e-9) throw new Error("이미지 범위 밖의 상자가 있습니다");
        }
      }
    }

    items.forEach(item => {
      const initial = {
        annotations:item.annotations.map(annotation => ({class_id:annotation.class_id,x_center:annotation.x_center,y_center:annotation.y_center,width:annotation.width,height:annotation.height,source:"original"})),
        note:item.review.note || "",
        excluded:item.review.status === "exclude",
        completed:false,
        updated_at:null
      };
      edits[item.image_path] = saved[item.image_path] || initial;
      history[item.image_path] = [];
    });

    function currentItem() { return items[current]; }
    function currentEdit() { return edits[currentItem().image_path]; }
    function cloneAnnotations() { return currentEdit().annotations.map(annotation => ({...annotation})); }
    function checkpoint() { history[currentItem().image_path].push({annotations:cloneAnnotations(),completed:currentEdit().completed}); }
    function save() {
      currentEdit().updated_at = new Date().toISOString();
      saved[currentItem().image_path]=currentEdit();
      try { localStorage.setItem(storageKey, JSON.stringify(saved));saveStatus("브라우저에 저장됨"); }
      catch(error) { saveStatus("자동 저장 실패 · JSON을 지금 내보내세요",true); }
    }
    function saveStatus(message,error=false) {const node=document.getElementById("save-status");node.textContent=message;node.classList.toggle("error",error);}
    function applyFilters() {
      const query=document.getElementById("search").value.toLowerCase();
      const split=document.getElementById("split-filter").value;
      const status=document.getElementById("status-filter").value;
      filtered=items.flatMap((item,index)=>{
        const edit=edits[item.image_path];
        const match=item.image_path.toLowerCase().includes(query) && (split==="all" || item.split===split) && (status==="all" || status==="pending" && !edit.completed || status==="completed" && edit.completed || status==="excluded" && edit.excluded || status==="empty" && edit.annotations.length===0);
        return match ? [index] : [];
      });
      if (filtered.length && !filtered.includes(current)) current=filtered[0];
      render();
    }

    function render() {
      renderList();
      const done=items.filter(item => edits[item.image_path].completed).length;
      document.getElementById("progress").textContent=`수정 완료 ${done} / ${items.length}`;
      document.querySelectorAll(".toolbar button,#complete,#complete-next,#exclude,#note").forEach(node=>node.disabled=!filtered.length);
      if(!filtered.length) {requestedPath=null;loadedPath=null;currentImage=null;canvas.width=1;canvas.height=1;document.getElementById("counter").textContent="0 / 0";document.getElementById("path").textContent="조건에 맞는 이미지가 없습니다";return;}
      const item=currentItem(); const edit=currentEdit();
      document.getElementById("counter").textContent=`${filtered.indexOf(current) + 1} / ${filtered.length}`;
      document.getElementById("path").textContent=item.image_path;
      document.getElementById("note").value=edit.note;
      document.getElementById("exclude").checked=edit.excluded;
      document.getElementById("meta").innerHTML=`<dt>분할</dt><dd>${escapeHtml(item.split)}</dd><dt>기존 라벨</dt><dd>${item.annotation_count}개</dd><dt>현재 라벨</dt><dd>${edit.annotations.length}개</dd><dt>원본 클래스</dt><dd>${escapeHtml(Object.entries(item.source_class_counts || {}).map(([key,value])=>`${key} ${value}개`).join(", ") || "정보 없음")}</dd>`;
      const complete=document.getElementById("complete");
      complete.textContent=edit.completed ? "수정 완료됨" : "이 이미지 수정 완료";
      complete.classList.toggle("done",edit.completed);
      loadImage(item);
    }

    function renderList() {
      const position=Math.max(0,filtered.indexOf(current));
      const start=Math.max(0,position-35);
      document.getElementById("filter-count").textContent=`필터 결과 ${filtered.length}장 · 목록 ${Math.min(start+1,filtered.length)}–${Math.min(start+80,filtered.length)}`;
      document.getElementById("item-list").innerHTML=filtered.slice(start,start+80).map(index => {
        const item=items[index];
        const edit=edits[item.image_path];
        const badge=edit.excluded ? "제외" : edit.completed ? "완료" : "대기";
        const badgeClass=edit.excluded ? "exclude" : edit.completed ? "done" : "";
        return `<button class="item ${index === current ? "active" : ""}" data-index="${index}"><div class="item-line"><span class="filename">${escapeHtml(item.image_path.split("/").pop())}</span><span class="badge ${badgeClass}">${badge}</span></div><div class="note-line">${item.split} · ${edit.annotations.length}개 · ${escapeHtml(edit.note || item.review.note || "편집 대기")}</div></button>`;
      }).join("");
      document.querySelectorAll(".item").forEach(button => button.addEventListener("click",() => {current=Number(button.dataset.index);selected=null;render();}));
    }

    function loadImage(item) {
      if(loadedPath===item.image_path && currentImage) {draw();return;}
      requestedPath=item.image_path;loadedPath=null;currentImage=null;gesture=null;dragStart=null;dragCurrent=null;
      context.clearRect(0,0,canvas.width,canvas.height);
      const image=new Image();
      image.onload=() => { if(requestedPath!==item.image_path) return;currentImage=image;loadedPath=item.image_path;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;resizeCanvas();draw(); };
      image.onerror=()=>{if(requestedPath===item.image_path) saveStatus("이미지를 불러오지 못했습니다",true);};
      image.src=item.asset_path;
    }

    function draw() {
      if (!currentImage) return;
      context.drawImage(currentImage,0,0);
      if(showBoxes) currentEdit().annotations.forEach((annotation,index) => drawBox(annotation,index,index === selected));
      if (dragStart && dragCurrent) {
        const annotation=boxFromPoints(dragStart,dragCurrent);
        drawBox({...annotation,source:"added"},-1,true);
      }
    }

    function drawBox(annotation,index,isSelected) {
      const x=(annotation.x_center-annotation.width/2)*canvas.width;
      const y=(annotation.y_center-annotation.height/2)*canvas.height;
      const width=annotation.width*canvas.width;
      const height=annotation.height*canvas.height;
      context.lineWidth=isSelected ? 4 : Math.max(2,canvas.width/320);
      context.strokeStyle=isSelected ? "#ffffff" : annotation.source === "added" ? "#ffd166" : "#55e3c2";
      context.strokeRect(x,y,width,height);
      if(isSelected) {const size=8*canvas.width/canvas.getBoundingClientRect().width;for(const [hx,hy] of [[x,y],[x+width,y],[x,y+height],[x+width,y+height]]){context.fillStyle="#fff";context.fillRect(hx-size/2,hy-size/2,size,size);}}
      context.font=`bold ${Math.max(12,canvas.width/45)}px sans-serif`;
      context.fillStyle="#071510cc";
      context.fillRect(x,Math.max(0,y-19),38,19);
      context.fillStyle=context.strokeStyle;
      context.fillText(index<0 ? "+" : `#${index + 1}`,x+3,Math.max(14,y-4));
    }

    function normalizedPoint(event) {
      const rectangle=canvas.getBoundingClientRect();
      return {x:Math.min(1,Math.max(0,(event.clientX-rectangle.left)/rectangle.width)),y:Math.min(1,Math.max(0,(event.clientY-rectangle.top)/rectangle.height))};
    }
    function boxFromPoints(first,second) {
      const left=Math.min(first.x,second.x); const right=Math.max(first.x,second.x);
      const top=Math.min(first.y,second.y); const bottom=Math.max(first.y,second.y);
      return {class_id:0,x_center:(left+right)/2,y_center:(top+bottom)/2,width:right-left,height:bottom-top};
    }
    function findBox(point) {
      let found=null; let smallest=Infinity;
      currentEdit().annotations.forEach((annotation,index) => {
        const inside=Math.abs(point.x-annotation.x_center)<=annotation.width/2 && Math.abs(point.y-annotation.y_center)<=annotation.height/2;
        const area=annotation.width*annotation.height;
        if (inside && area<smallest) {found=index;smallest=area;}
      });
      return found;
    }

    function resizeCanvas() {
      if(!currentImage) return;
      const viewer=document.querySelector(".viewer");
      const fit=Math.min((viewer.clientWidth-28)/canvas.width,(viewer.clientHeight-28)/canvas.height,1);
      canvas.style.width=`${Math.max(1,canvas.width*fit*Number(document.getElementById("zoom").value))}px`;
      canvas.style.height="auto";
    }
    function selectedHandle(point) {
      if(selected===null) return null;
      const box=currentEdit().annotations[selected];
      if(!box) return null;
      const rect=canvas.getBoundingClientRect();
      for(const [x,y] of [[-1,-1],[1,-1],[-1,1],[1,1]]) {
        const corner={x:box.x_center+x*box.width/2,y:box.y_center+y*box.height/2};
        if(Math.abs(point.x-corner.x)*rect.width<=10 && Math.abs(point.y-corner.y)*rect.height<=10) return {x:box.x_center-x*box.width/2,y:box.y_center-y*box.height/2};
      }
      return null;
    }
    canvas.addEventListener("pointerdown",event=>{
      if(!currentImage || loadedPath!==currentItem().image_path || !filtered.length) return;
      event.preventDefault();canvas.setPointerCapture(event.pointerId);
      const point=normalizedPoint(event);
      if(mode==="draw") {dragStart=point;dragCurrent=point;return;}
      const anchor=selectedHandle(point);
      if(!anchor) selected=findBox(point);
      if(selected!==null) gesture={start:point,anchor,box:{...currentEdit().annotations[selected]},snapshot:cloneAnnotations(),completed:currentEdit().completed,moved:false};
      draw();
    });
    canvas.addEventListener("pointermove",event=>{
      const point=normalizedPoint(event);
      if(dragStart) {dragCurrent=point;draw();return;}
      if(!gesture) return;
      const before=gesture.box;
      let box;
      if(gesture.anchor) {box={...before,...boxFromPoints(gesture.anchor,point)};if(box.width<0.001 || box.height<0.001) return;}
      else {box={...before,x_center:Math.min(1-before.width/2,Math.max(before.width/2,before.x_center+point.x-gesture.start.x)),y_center:Math.min(1-before.height/2,Math.max(before.height/2,before.y_center+point.y-gesture.start.y))};}
      gesture.moved=JSON.stringify(box)!==JSON.stringify(before);
      currentEdit().annotations[selected]=box;draw();
    });
    canvas.addEventListener("pointerup",event=>{
      if(dragStart) {
        const box=boxFromPoints(dragStart,normalizedPoint(event));dragStart=null;dragCurrent=null;
        if(box.width*canvas.width>=2 && box.height*canvas.height>=2) {checkpoint();currentEdit().annotations.push({...box,source:"added"});selected=currentEdit().annotations.length-1;currentEdit().completed=false;save();}
      }
      if(gesture) {
        if(gesture.moved) {history[currentItem().image_path].push({annotations:gesture.snapshot,completed:gesture.completed});currentEdit().completed=false;save();}
        gesture=null;
      }
      render();
    });
    canvas.addEventListener("pointercancel",()=>{if(gesture) currentEdit().annotations=gesture.snapshot;gesture=null;dragStart=null;dragCurrent=null;draw();});

    function setMode(nextMode) {mode=nextMode;showBoxes=true;document.getElementById("toggle-boxes").textContent="상자 숨기기";document.getElementById("draw-mode").classList.toggle("active",mode === "draw");document.getElementById("select-mode").classList.toggle("active",mode === "select");canvas.style.cursor=mode === "draw" ? "crosshair" : "pointer";draw();}
    function move(offset) {if(!filtered.length) return;current=filtered[(filtered.indexOf(current)+offset+filtered.length)%filtered.length];selected=null;render();}
    function deleteSelected() {if (selected === null || !filtered.length) return;checkpoint();currentEdit().annotations.splice(selected,1);currentEdit().completed=false;selected=null;save();render();}
    function undo() {if(!filtered.length) return;const previous=history[currentItem().image_path].pop();if (!previous) return;currentEdit().annotations=previous.annotations;currentEdit().completed=previous.completed;selected=null;save();render();}
    function escapeHtml(value) {const node=document.createElement("span");node.textContent=value;return node.innerHTML;}

    document.getElementById("draw-mode").addEventListener("click",() => setMode("draw"));
    document.getElementById("select-mode").addEventListener("click",() => setMode("select"));
    document.getElementById("delete-box").addEventListener("click",deleteSelected);
    document.getElementById("undo").addEventListener("click",undo);
    document.getElementById("previous").addEventListener("click",() => move(-1));
    document.getElementById("next").addEventListener("click",() => move(1));
    document.getElementById("note").addEventListener("input",event => {currentEdit().note=event.target.value;save();renderList();});
    document.getElementById("exclude").addEventListener("change",event => {currentEdit().excluded=event.target.checked;currentEdit().completed=false;save();render();});
    document.getElementById("complete").addEventListener("click",() => {currentEdit().completed=!currentEdit().completed;save();render();});
    document.getElementById("reset").addEventListener("click",() => {checkpoint();currentEdit().annotations=currentItem().annotations.map(annotation => ({class_id:annotation.class_id,x_center:annotation.x_center,y_center:annotation.y_center,width:annotation.width,height:annotation.height,source:"original"}));currentEdit().completed=false;selected=null;save();render();});
    document.getElementById("complete-next").addEventListener("click",()=>{currentEdit().completed=true;save();move(1);applyFilters();});
    document.getElementById("toggle-boxes").addEventListener("click",event=>{showBoxes=!showBoxes;event.target.textContent=showBoxes ? "상자 숨기기" : "상자 표시";draw();});
    document.getElementById("zoom").addEventListener("change",resizeCanvas);
    window.addEventListener("resize",resizeCanvas);
    for(const id of ["search","split-filter","status-filter"]) document.getElementById(id).addEventListener(id==="search" ? "input" : "change",()=>{selected=null;applyFilters();});
    document.getElementById("import-file").addEventListener("change",async event=>{
      const file=event.target.files[0];if(!file) return;
      try {
        const payload=JSON.parse(await file.text());
        if(payload.schema_version!==1 || payload.dataset_fingerprint!==datasetFingerprint) throw new Error("현재 데이터와 다른 편집 결과입니다");
        validateEdits(payload.edits);
        if(Object.keys(payload.edits).some(path=>saved[path]) && !confirm("현재 작업과 겹치는 이미지가 있습니다. 해당 이미지의 작업을 불러온 결과로 바꿀까요?")) return;
        const merged={...saved,...payload.edits};
        localStorage.setItem(storageKey,JSON.stringify(merged));saved=merged;
        for(const [path,edit] of Object.entries(payload.edits)) {edits[path]=edit;history[path]=[];}
        selected=null;saveStatus("편집 결과를 불러왔습니다");applyFilters();
      } catch(error) {alert(`불러오기 실패: ${error.message}`);}
      finally {event.target.value="";}
    });
    document.getElementById("export").addEventListener("click",()=>{
      const payload={schema_version:1,dataset_fingerprint:datasetFingerprint,review_splits:allItems ? ["train","valid","test"] : ["valid","test"],exported_at:new Date().toISOString(),edits};
      const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=allItems ? "acne-care-external-label-edits.json" : "acne-care-label-edits.json";link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
    });
    document.addEventListener("keydown",event=>{if(event.target.matches("textarea,input,select") || !filtered.length) return;if(["ArrowLeft","ArrowRight","Delete","Backspace"].includes(event.key)) event.preventDefault();if(event.key==="ArrowLeft") move(-1);if(event.key==="ArrowRight") move(1);if(event.key==="Delete" || event.key==="Backspace") deleteSelected();if(event.key.toLowerCase()==="d") setMode("draw");if(event.key.toLowerCase()==="s") setMode("select");});
    if(allItems) document.getElementById("dataset-help").prepend("acne·pimple만 여드름 상자로 표시합니다. 평평한 자국인 spot은 제외했습니다. 상자 없는 이미지도 확인 후 완료를 눌러주세요. ");
    render();
    if(startupError) saveStatus(startupError,true);
  </script>
</body>
</html>
"""
