from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACTIONABLE_STATUSES = {"needs_correction", "exclude"}


def render_label_editor_html(
    document: dict[str, Any],
    review_results: dict[str, Any],
) -> str:
    reviews = review_results.get("reviews")
    if not isinstance(reviews, dict):
        raise ValueError("review results must contain a reviews object")
    items = []
    for source_item in document["items"]:
        review = reviews.get(source_item["image_path"], {})
        if review.get("status") not in ACTIONABLE_STATUSES:
            continue
        item = dict(source_item)
        item["asset_path"] = (
            Path("assets") / item["split"] / Path(item["image_path"]).name
        ).as_posix()
        item["review"] = review
        items.append(item)
    serialized = json.dumps(items, ensure_ascii=False).replace("<", "\\u003c")
    fingerprint = str(review_results.get("dataset_fingerprint", "unknown"))
    return TEMPLATE.replace("__EDITOR_ITEMS__", serialized).replace(
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
    button,input,textarea { font:inherit; }
    button { border:1px solid var(--line); border-radius:8px; padding:8px 11px; color:var(--text); background:#1d2845; cursor:pointer; }
    button:hover,button.active { border-color:var(--mint); }
    header { height:66px; display:flex; align-items:center; gap:18px; padding:14px 20px; background:#10172a; border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:19px; }
    #progress { color:var(--muted); }
    #export { margin-left:auto; }
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
    .viewer { flex:1; min-height:0; display:grid; place-items:center; overflow:auto; padding:14px; }
    canvas { max-width:100%; max-height:100%; object-fit:contain; box-shadow:0 12px 40px #000a; cursor:crosshair; touch-action:none; }
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
    @media (max-width:1050px) { .layout { grid-template-columns:260px minmax(0,1fr); } .inspector { position:fixed; right:0; top:66px; bottom:0; width:300px; box-shadow:-10px 0 30px #0009; } main { margin-right:300px; } }
  </style>
</head>
<body>
  <header><h1>acne-care 라벨 편집</h1><span id="progress"></span><button id="export">편집 결과 내보내기</button></header>
  <div class="layout">
    <aside id="item-list"></aside>
    <main>
      <div class="viewer-head"><strong id="counter"></strong><span id="path"></span></div>
      <div class="viewer"><canvas id="canvas"></canvas></div>
      <div class="toolbar">
        <button id="previous">← 이전</button>
        <button id="select-mode">상자 선택</button>
        <button id="draw-mode" class="active">상자 추가</button>
        <button id="delete-box" class="danger">선택 상자 삭제</button>
        <button id="undo">되돌리기</button>
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
      <label class="exclude-row"><input id="exclude" type="checkbox"> dataset-v2에서 이미지 제외</label>
      <button id="complete">이 이미지 수정 완료</button>
      <p class="help">상자 추가 모드에서 병변을 드래그하면 새 상자가 생깁니다. 잘못된 상자는 선택한 다음 삭제할 수 있습니다. 기존 상자의 위치나 크기를 바꾸려면 삭제하고 다시 그립니다. 모든 변경은 브라우저에 자동 저장되며 원본 라벨은 바뀌지 않습니다.</p>
    </section>
  </div>
  <script>
    const items = __EDITOR_ITEMS__;
    const datasetFingerprint = "__DATASET_FINGERPRINT__";
    const storageKey = `acne-care-label-editor-${datasetFingerprint}`;
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
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
    function checkpoint() { history[currentItem().image_path].push(cloneAnnotations()); }
    function save() { currentEdit().updated_at = new Date().toISOString(); localStorage.setItem(storageKey, JSON.stringify(edits)); }

    function render() {
      renderList();
      const item=currentItem(); const edit=currentEdit();
      document.getElementById("counter").textContent=`${current + 1} / ${items.length}`;
      document.getElementById("path").textContent=item.image_path;
      document.getElementById("note").value=edit.note;
      document.getElementById("exclude").checked=edit.excluded;
      document.getElementById("meta").innerHTML=`<dt>분할</dt><dd>${item.split}</dd><dt>기존 라벨</dt><dd>${item.annotation_count}개</dd><dt>현재 라벨</dt><dd>${edit.annotations.length}개</dd><dt>검토 상태</dt><dd>${item.review.status === "exclude" ? "제외 후보" : "수정 필요"}</dd>`;
      const complete=document.getElementById("complete");
      complete.textContent=edit.completed ? "수정 완료됨" : "이 이미지 수정 완료";
      complete.classList.toggle("done",edit.completed);
      const done=items.filter(item => edits[item.image_path].completed).length;
      document.getElementById("progress").textContent=`수정 완료 ${done} / ${items.length}`;
      loadImage(item);
    }

    function renderList() {
      document.getElementById("item-list").innerHTML=items.map((item,index) => {
        const edit=edits[item.image_path];
        const badge=edit.excluded ? "제외" : edit.completed ? "완료" : "대기";
        const badgeClass=edit.excluded ? "exclude" : edit.completed ? "done" : "";
        return `<button class="item ${index === current ? "active" : ""}" data-index="${index}"><div class="item-line"><span class="filename">${escapeHtml(item.image_path.split("/").pop())}</span><span class="badge ${badgeClass}">${badge}</span></div><div class="note-line">${escapeHtml(item.review.note || "검토 메모 없음")}</div></button>`;
      }).join("");
      document.querySelectorAll(".item").forEach(button => button.addEventListener("click",() => {current=Number(button.dataset.index);selected=null;render();}));
    }

    function loadImage(item) {
      const image=new Image();
      image.onload=() => { currentImage=image; canvas.width=image.naturalWidth; canvas.height=image.naturalHeight; draw(); };
      image.src=item.asset_path;
    }

    function draw() {
      if (!currentImage) return;
      context.drawImage(currentImage,0,0);
      currentEdit().annotations.forEach((annotation,index) => drawBox(annotation,index,index === selected));
      if (dragStart && dragCurrent) {
        const annotation=boxFromPoints(dragStart,dragCurrent);
        drawBox({...annotation,source:"added"},"+",true);
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
      context.font=`bold ${Math.max(12,canvas.width/45)}px sans-serif`;
      context.fillStyle="#071510cc";
      context.fillRect(x,Math.max(0,y-19),38,19);
      context.fillStyle=context.strokeStyle;
      context.fillText(`#${index + 1}`,x+3,Math.max(14,y-4));
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

    canvas.addEventListener("pointerdown",event => { canvas.setPointerCapture(event.pointerId); if (mode === "draw") {dragStart=normalizedPoint(event);dragCurrent=dragStart;} else {selected=findBox(normalizedPoint(event));draw();} });
    canvas.addEventListener("pointermove",event => {if (dragStart) {dragCurrent=normalizedPoint(event);draw();}});
    canvas.addEventListener("pointerup",event => {if (!dragStart) return; const box=boxFromPoints(dragStart,normalizedPoint(event));dragStart=null;dragCurrent=null;if (box.width>=0.005 && box.height>=0.005) {checkpoint();currentEdit().annotations.push({...box,source:"added"});selected=currentEdit().annotations.length-1;save();}render();});

    function setMode(nextMode) {mode=nextMode;document.getElementById("draw-mode").classList.toggle("active",mode === "draw");document.getElementById("select-mode").classList.toggle("active",mode === "select");canvas.style.cursor=mode === "draw" ? "crosshair" : "pointer";}
    function move(offset) {current=(current+offset+items.length)%items.length;selected=null;render();}
    function deleteSelected() {if (selected === null) return;checkpoint();currentEdit().annotations.splice(selected,1);selected=null;save();render();}
    function undo() {const previous=history[currentItem().image_path].pop();if (!previous) return;currentEdit().annotations=previous;selected=null;save();render();}
    function escapeHtml(value) {const node=document.createElement("span");node.textContent=value;return node.innerHTML;}

    document.getElementById("draw-mode").addEventListener("click",() => setMode("draw"));
    document.getElementById("select-mode").addEventListener("click",() => setMode("select"));
    document.getElementById("delete-box").addEventListener("click",deleteSelected);
    document.getElementById("undo").addEventListener("click",undo);
    document.getElementById("previous").addEventListener("click",() => move(-1));
    document.getElementById("next").addEventListener("click",() => move(1));
    document.getElementById("note").addEventListener("input",event => {currentEdit().note=event.target.value;save();renderList();});
    document.getElementById("exclude").addEventListener("change",event => {currentEdit().excluded=event.target.checked;save();render();});
    document.getElementById("complete").addEventListener("click",() => {currentEdit().completed=!currentEdit().completed;save();render();});
    document.getElementById("reset").addEventListener("click",() => {checkpoint();currentEdit().annotations=currentItem().annotations.map(annotation => ({class_id:annotation.class_id,x_center:annotation.x_center,y_center:annotation.y_center,width:annotation.width,height:annotation.height,source:"original"}));currentEdit().completed=false;selected=null;save();render();});
    document.getElementById("export").addEventListener("click",() => {const payload={schema_version:1,dataset_fingerprint:datasetFingerprint,exported_at:new Date().toISOString(),edits};const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="acne-care-label-edits.json";link.click();URL.revokeObjectURL(link.href);});
    document.addEventListener("keydown",event => {if (event.target.matches("textarea,input")) return;if (event.key === "ArrowLeft") move(-1);if (event.key === "ArrowRight") move(1);if (event.key === "Delete" || event.key === "Backspace") deleteSelected();if (event.key.toLowerCase() === "d") setMode("draw");if (event.key.toLowerCase() === "s") setMode("select");});
    render();
  </script>
</body>
</html>
"""
