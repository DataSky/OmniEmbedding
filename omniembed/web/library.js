// 视频库 + 详情（元数据、抽帧、向量数据、相似视频、编辑）
const LIB = { offset: 0, q: "" };
function card(v, i) {
  const sc = v.scores ? Object.entries(v.scores).map(([k, x]) => `${k} ${x}`).join(" · ") : `${STLABEL[v.status] || ""} · ${fmtS(v.duration_s)}`;
  const rk = v.rank ? `<span class="rk">${v.rank}</span>` : "";
  return `<div class="card" tabindex="0" role="button" data-open="${esc(v.video_id || v.id)}">
    <div class="cv" style="${v.cover_url ? `background-image:url('${esc(v.cover_url)}')` : ""}">${rk}</div>
    <div class="bd"><div class="tt">${esc(v.title || v.source_ref || "(无标题)")}</div><div class="sc">${esc(sc)}</div></div></div>`;
}
function bindCards(root) {
  root.querySelectorAll(".card").forEach(c => {
    c.onclick = () => openDetail(c.dataset.open);
    c.onkeydown = e => { if (e.key === "Enter") openDetail(c.dataset.open); };
  });
}
async function loadLibrary(reset = true) {
  if (reset) { LIB.offset = 0; $("#lib-grid").innerHTML = ""; }
  const r = await api(`/api/videos?status=ready&limit=48&offset=${LIB.offset}` + (LIB.q ? "&q=" + encodeURIComponent(LIB.q) : ""));
  $("#lib-grid").insertAdjacentHTML("beforeend", r.items.map(card).join(""));
  if (!r.total) $("#lib-grid").innerHTML = `<p class="mute">视频库是空的。到「入库」上传视频或提交地址，处理完成后会出现在这里。</p>`;
  LIB.offset += r.items.length;
  $("#lib-count").textContent = `共 ${r.total} 条已入库`;
  $("#lib-more").hidden = LIB.offset >= r.total;
  bindCards($("#lib-grid"));
}
let libTimer;
$("#lib-q").oninput = e => { clearTimeout(libTimer); libTimer = setTimeout(() => { LIB.q = e.target.value.trim(); loadLibrary(); }, 250); };
$("#lib-more").onclick = () => loadLibrary(false);
TABS.library = () => loadLibrary();

function spark(canvas, values) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth || 400, h = canvas.clientHeight || 60;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d"); g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const m = Math.max(...values.map(Math.abs), 1e-6), bw = w / values.length;
  g.fillStyle = css.getPropertyValue("--line"); g.fillRect(0, h / 2, w, 1);
  g.fillStyle = css.getPropertyValue("--acc");
  values.forEach((v, i) => { const y = (v / m) * (h / 2 - 2); g.fillRect(i * bw, v >= 0 ? h / 2 - y : h / 2, Math.max(bw - 0.3, 0.5), Math.abs(y)); });
}

async function openDetail(id) {
  const dlg = $("#detail");
  $("#d-body").innerHTML = `<div class="dh">加载中…</div>`;
  if (!dlg.open) dlg.showModal();
  const v = await api(`/api/videos/${encodeURIComponent(id)}`);
  const vk = Object.keys(v.vectors);
  const kv = (k, x) => `<dt>${esc(k)}</dt><dd>${x}</dd>`;
  const meta = [
    kv("ID", `<code>${esc(v.id)}</code>`), kv("状态", st(v.status) + (v.error ? ` <span class="err">${esc(v.error)}</span>` : "")),
    kv("来源", `${esc(v.source)} · ${esc(v.source_ref)}`), kv("时长", fmtS(v.duration_s)),
    kv("分辨率", v.width ? `${v.width}×${v.height}` : "—"), kv("帧率", v.fps ?? "—"), kv("编码", esc(v.codec || "—")),
    kv("音轨", v.has_audio == null ? "—" : v.has_audio ? "有" : "无"), kv("大小", fmtMB(v.bytes)),
    kv("SHA-256", `<code style="font-size:11px">${esc((v.sha256 || "").slice(0, 16))}…</code>`),
    kv("各阶段耗时", esc(Object.entries(v.timings || {}).map(([k, x]) => `${k} ${x}`).join(" · ") || "—")),
    kv("入库时间", new Date(v.created_at * 1000).toLocaleString()),
    kv("业务字段", `<code style="font-size:11px">${esc(JSON.stringify(v.extra || {}))}</code>`),
  ].join("");
  const frames = v.frames.map(f => `<figure><img src="${esc(f.thumb_url)}" alt="第 ${f.idx + 1} 帧" loading="lazy"><figcaption>${f.t_s}s</figcaption></figure>`).join("");
  $("#d-body").innerHTML = `
    <div class="dh"><b id="d-title">${esc(v.title || "(无标题)")}</b><button type="button" onclick="detail.close()">关闭</button></div>
    <div class="db">
      <div>
        ${v.video_url ? `<video src="${esc(v.video_url)}" controls playsinline preload="metadata"></video>` : ""}
        <h3>抽帧（${v.frames.length} 帧，用于画面向量）</h3><div class="frames">${frames || '<span class="mute">无</span>'}</div>
        <h3>编辑元数据 <span class="mute">保存后重算标题向量与融合向量，不重新抽帧</span></h3>
        <label class="fld">标题<input id="e-title" value="${esc(v.title)}"></label>
        <label class="fld">描述<textarea id="e-desc" rows="2">${esc(v.description)}</textarea></label>
        <label class="fld">标签<input id="e-tags" value="${esc((v.tags || []).join(", "))}"></label>
        <div class="row"><button id="e-save" class="primary" type="button">保存</button>
          ${v.status === "failed" ? '<button id="e-retry" type="button">重试</button>' : ""}
          <button id="e-del" type="button">删除</button><span id="e-out" class="mute" aria-live="polite"></span></div>
      </div>
      <div>
        <h3>元数据</h3><dl class="kv">${meta}</dl>
        <h3>向量数据</h3>
        ${vk.length ? `<div class="row" role="group" aria-label="向量类型">${vk.map(k => `<button type="button" data-vk="${esc(k)}">${esc(k)}</button>`).join("")}</div>
          <div id="vec-box"></div>` : '<p class="mute">还没有向量（处理未完成或失败）。</p>'}
        <h3>画面相似的视频</h3><div id="d-sim" class="grid" style="grid-template-columns:repeat(auto-fill,minmax(130px,1fr))"></div>
      </div>
    </div>`;
  $("#e-save").onclick = async () => {
    try {
      await api(`/api/videos/${v.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: $("#e-title").value, description: $("#e-desc").value, tags: $("#e-tags").value }) });
      $("#e-out").textContent = "已保存，向量已更新。";
    } catch (e) { $("#e-out").textContent = "保存失败：" + e.message; }
  };
  if ($("#e-retry")) $("#e-retry").onclick = async () => { await api(`/api/videos/${v.id}/retry`, { method: "POST" }); $("#e-out").textContent = "已重新排队。"; };
  // 删除需要二次确认（页面内确认，不依赖浏览器 confirm）
  $("#e-del").onclick = async e => {
    const b = e.currentTarget;
    if (b.dataset.armed !== "1") { b.dataset.armed = "1"; b.textContent = "再点一次确认删除（同时删除视频文件）"; return; }
    await api(`/api/videos/${v.id}`, { method: "DELETE" });
    detail.close(); refreshStat(); if (!$("#tab-library").hidden) loadLibrary();
  };
  const showVec = async kind => {
    document.querySelectorAll("[data-vk]").forEach(b => b.setAttribute("aria-pressed", b.dataset.vk === kind));
    const r = await api(`/api/videos/${v.id}/vector?kind=${kind}&full=true`);
    const s = r.stats;
    $("#vec-box").innerHTML = `<dl class="kv">${kv("维度 / 模型", `${r.dim} · ${esc(r.model)}`)}${kv("模长", s.norm)}${kv("均值 / 标准差", `${s.mean} / ${s.std}`)}
      ${kv("最小 / 最大", `${s.min} / ${s.max}`)}${kv("与其他向量余弦", esc(Object.entries(r.cosine_with).map(([k, x]) => `${k} ${x}`).join(" · ") || "—"))}</dl>
      <canvas class="spark" aria-label="768 维向量取值分布"></canvas>
      <div class="vec" tabindex="0">[${r.values.join(", ")}]</div>
      <div class="row" style="margin-top:6px"><button id="vec-copy" type="button">复制向量</button><span id="vec-cp" class="mute"></span></div>`;
    spark($("#vec-box canvas"), r.values);
    $("#vec-copy").onclick = async () => {
      try { await navigator.clipboard.writeText(JSON.stringify(r.values)); $("#vec-cp").textContent = "已复制"; }
      catch { const sel = getSelection(); sel.selectAllChildren($("#vec-box .vec")); $("#vec-cp").textContent = "已选中，按 Ctrl+C 复制"; }
    };
  };
  document.querySelectorAll("[data-vk]").forEach(b => b.onclick = () => showVec(b.dataset.vk));
  if (vk.length) showVec(vk.includes("visual") ? "visual" : vk[0]);
  if (v.status === "ready") {
    const r = await api(`/api/videos/${v.id}/similar?k=6`);
    $("#d-sim").innerHTML = r.results.map(card).join("") || '<span class="mute">库里还没有其他视频</span>';
    bindCards($("#d-sim"));
  }
}
$("#detail").addEventListener("close", () => $("#detail video")?.pause());
