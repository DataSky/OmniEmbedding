// 公共工具 + 标签页 + 入库 + 任务列表
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const KEY = () => { try { return localStorage.getItem("omni_api_key") || ""; } catch { return ""; } };
async function api(path, opts = {}) {
  const h = Object.assign({}, opts.headers || {});
  if (KEY()) h["X-API-Key"] = KEY();
  const r = await fetch(path, Object.assign({}, opts, { headers: h }));
  const body = r.headers.get("content-type")?.includes("json") ? await r.json() : await r.text();
  if (r.status === 401) showKeyBox();  // 用页面内输入框，不依赖浏览器 prompt
  if (!r.ok) throw new Error(body?.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `HTTP ${r.status}`);
  return body;
}
function showKeyBox() {
  if ($("#keybox")) return;
  const d = document.createElement("div");
  d.id = "keybox"; d.className = "panel"; d.style.margin = "0 0 12px";
  d.innerHTML = `服务开启了访问密钥。<input id="kb-in" type="password" placeholder="X-API-Key" aria-label="访问密钥"> <button id="kb-ok" type="button">保存</button>`;
  $("main").prepend(d);
  $("#kb-ok").onclick = () => { try { localStorage.setItem("omni_api_key", $("#kb-in").value.trim()); } catch {} d.remove(); };
}
const fmtS = s => s == null ? "—" : s < 60 ? `${(+s).toFixed(1)}s` : `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
const fmtMB = b => b == null ? "—" : `${(b / 2 ** 20).toFixed(1)} MiB`;
const STLABEL = { queued: "排队", downloading: "下载", probing: "解析", extracting: "抽帧", embedding: "向量化", ready: "完成", failed: "失败" };
const st = s => `<span class="st ${esc(s)}"><i></i>${esc(STLABEL[s] || s)}</span>`;

// ---------- 标签页 ----------
const TABS = {};
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => showTab(b.dataset.tab));
function showTab(name) {
  $("#stat").hidden = name === "health";
  document.querySelectorAll(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(s => s.hidden = s.id !== "tab-" + name);
  try { history.replaceState(null, "", "#" + name); } catch {}
  TABS[name]?.();
}

// ---------- 状态栏 ----------
async function refreshStat() {
  try {
    const s = await api("/api/stats");
    $("#ver").textContent = "v" + s.version;
    const c = s.status || {}, active = ["queued", "downloading", "probing", "extracting", "embedding"].reduce((a, k) => a + (c[k] || 0), 0);
    $("#jobbadge").textContent = active || "";
    const encoder = s.encoder_backend === "remote"
      ? `Colab 远程推理${s.remote_last_success_at ? " · 最近调用 " + new Date(s.remote_last_success_at).toLocaleTimeString("zh-CN", { timeZone: "Asia/Shanghai", hour12: false }) : " · 等待首次调用"}`
      : `本机模型 ${s.model_loaded ? s.device : "未加载"}`;
    $("#stat").textContent = `已入库 ${s.indexed} · 处理中 ${active} · 失败 ${c.failed || 0} · ${encoder} · 抽帧 ${s.frames.fps}fps ≤${s.frames.max_frames} 帧`;
    return active;
  } catch (e) { $("#stat").textContent = "服务不可用：" + e.message; return 0; }
}

// ---------- 入库 ----------
let picked = [];
const drop = $("#drop");
$("#files").onchange = e => { picked = [...e.target.files]; drop.textContent = `已选 ${picked.length} 个文件`; };
["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { picked = [...e.dataTransfer.files]; drop.textContent = `已选 ${picked.length} 个文件`; });
$("#up-go").onclick = async () => {
  if (!picked.length) { $("#up-out").textContent = "先选择视频文件。"; return; }
  const btn = $("#up-go"); btn.disabled = true;
  let ok = 0, errs = [];
  // 分批上传，单批 10 个，避免一次请求过大
  for (let i = 0; i < picked.length; i += 10) {
    const fd = new FormData();
    picked.slice(i, i + 10).forEach(f => fd.append("files", f));
    fd.append("description", $("#up-desc").value); fd.append("tags", $("#up-tags").value);
    fd.append("filename_as_title", $("#up-fname").checked ? "true" : "false");
    $("#up-out").textContent = `上传中 ${Math.min(i + 10, picked.length)} / ${picked.length}…`;
    try { const r = await api("/api/videos/upload", { method: "POST", body: fd }); ok += r.queued.length; errs.push(...r.errors); }
    catch (e) { errs.push({ file: `第 ${i + 1}–${i + 10} 个`, error: e.message }); }
  }
  btn.disabled = false;
  $("#up-out").textContent = `已加入队列 ${ok} 个` + (errs.length ? `；跳过 ${errs.length} 个：\n` + errs.map(x => `${x.file}：${x.error}`).join("\n") : "。到「任务」查看进度。");
  picked = []; drop.textContent = "拖入视频，或点击选择（可多选）"; refreshStat();
};
$("#url-go").onclick = async () => {
  const fd = new FormData();
  fd.append("urls", $("#url-text").value);
  if ($("#url-file").files[0]) fd.append("manifest", $("#url-file").files[0]);
  try {
    const r = await api("/api/videos/urls", { method: "POST", body: fd });
    $("#url-out").textContent = `已加入队列 ${r.queued.length} 条` + (r.errors.length ? `；跳过 ${r.errors.length} 条：\n` + r.errors.map(x => `${x.url}：${x.error}`).join("\n") : "。");
    $("#url-text").value = ""; $("#url-file").value = ""; refreshStat();
  } catch (e) { $("#url-out").textContent = "提交失败：" + e.message; }
};
$("#dir-go").onclick = async () => {
  try {
    const r = await api("/api/videos/folder", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("#dir-path").value.trim(), limit: +$("#dir-limit").value || 200 }) });
    $("#dir-out").textContent = `找到 ${r.found} 个视频，已加入队列 ${r.queued.length} 个。`; refreshStat();
  } catch (e) { $("#dir-out").textContent = "导入失败：" + e.message; }
};

// ---------- 任务 ----------
async function loadJobs() {
  const s = $("#job-status").value;
  const r = await api("/api/videos?limit=200" + (s ? "&status=" + s : ""));
  const c = (await api("/api/stats")).status || {};
  $("#job-counts").textContent = Object.entries(c).map(([k, v]) => `${STLABEL[k] || k} ${v}`).join(" · ");
  $("#job-rows").innerHTML = r.items.map(v => {
    const t = v.timings || {}, total = ["download_s", "probe_s", "extract_s", "embed_s"].reduce((a, k) => a + (t[k] || 0), 0);
    return `<tr><td><a href="#" data-open="${esc(v.id)}">${esc(v.title || v.source_ref)}</a><div class="mute">${esc(v.id)}</div></td>
      <td>${esc(v.source)}<div class="mute" style="max-width:260px;overflow-wrap:anywhere">${esc(v.source_ref)}</div></td>
      <td>${st(v.status)}</td><td class="num">${fmtS(v.duration_s)}</td>
      <td class="num" title="${esc(JSON.stringify(t))}">${total ? fmtS(total) : "—"}</td>
      <td class="err">${esc(v.error || "")}</td>
      <td>${v.status === "failed" ? `<button type="button" data-retry="${esc(v.id)}">重试</button>` : ""}</td></tr>`;
  }).join("") || `<tr><td colspan="7" class="mute">还没有任务。到「入库」上传视频或提交地址。</td></tr>`;
  $("#job-rows").querySelectorAll("[data-retry]").forEach(b => b.onclick = async () => { await api(`/api/videos/${b.dataset.retry}/retry`, { method: "POST" }); loadJobs(); });
  $("#job-rows").querySelectorAll("[data-open]").forEach(a => a.onclick = e => { e.preventDefault(); openDetail(a.dataset.open); });
}
$("#job-status").onchange = loadJobs;
TABS.jobs = loadJobs;

// 有任务在跑时每 3 秒刷新一次
setInterval(async () => {
  const active = await refreshStat();
  if (active && !$("#tab-jobs").hidden) loadJobs();
}, 3000);
refreshStat();
