// 召回 + 评估
const MODE_LABEL = { hybrid: "混合", dense: "融合向量", visual: "仅画面", text: "仅标题向量", bm25: "仅 BM25" };

$("#s-form").onsubmit = async e => {
  e.preventDefault();
  const q = $("#s-q").value.trim(); if (!q) return;
  $("#s-meta").textContent = "检索中…";
  try {
    const r = await api(`/api/search?` + new URLSearchParams({ q, mode: $("#s-mode").value, dim: $("#s-dim").value, k: 30 }));
    const t = r.timing;
    $("#s-meta").textContent = `${r.results.length} 条 · ${MODE_LABEL[r.mode]} · ${r.dim} 维 · 查询编码 ${t.encode_ms ?? 0}ms · 打分 ${t.score_ms ?? 0}ms · 总计 ${t.total_ms}ms`;
    $("#s-grid").innerHTML = r.results.map(card).join("") || `<p class="mute">没有结果。库里是否已有处理完成的视频？</p>`;
    bindCards($("#s-grid"));
  } catch (err) { $("#s-meta").textContent = "检索失败：" + err.message; }
};
$("#s-img").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  $("#s-meta").textContent = "以图搜视频中…";
  try {
    const r = await api("/api/search/image?k=30", { method: "POST", body: fd });
    $("#s-meta").textContent = `以图搜视频：${r.results.length} 条（画面向量）`;
    $("#s-grid").innerHTML = r.results.map(card).join(""); bindCards($("#s-grid"));
  } catch (err) { $("#s-meta").textContent = "失败：" + err.message; }
  e.target.value = "";
};

async function loadEvalsets() {
  const r = await api("/api/evalsets");
  $("#ev-set").innerHTML = r.items.map(x => `<option value="${esc(x.name)}">${esc(x.name)}（${x.n} 条）</option>`).join("")
    || `<option value="">还没有评测集</option>`;
}
TABS.eval = loadEvalsets;

async function putEvalset(fd) {
  try {
    const r = await api("/api/evalsets", { method: "POST", body: fd });
    $("#ev-out").textContent = `已保存「${r.name}」：${r.n} 条` + (r.dropped ? `，丢弃 ${r.dropped} 条（query 为空或找不到对应视频）` : "");
    await loadEvalsets(); $("#ev-set").value = r.name;
  } catch (e) { $("#ev-out").textContent = "导入失败：" + e.message; }
}
$("#ev-up").onclick = () => {
  const f = $("#ev-file").files[0];
  if (!f) { $("#ev-out").textContent = "先选择评测集文件。"; return; }
  const fd = new FormData(); fd.append("name", $("#ev-name").value || f.name.replace(/\.\w+$/, "")); fd.append("file", f);
  putEvalset(fd);
};
$("#ev-auto").onclick = () => {
  const fd = new FormData(); fd.append("name", $("#ev-name").value || "auto-title"); fd.append("auto", "title");
  putEvalset(fd);
};
$("#ev-run").onclick = async () => {
  const name = $("#ev-set").value; if (!name) return;
  const btn = $("#ev-run"); btn.disabled = true;
  $("#ev-res").innerHTML = `<p class="mute">运行中…</p>`;
  try {
    const r = await api(`/api/evalsets/${encodeURIComponent(name)}/run?dim=${$("#ev-dim").value}`, { method: "POST" });
    const best = Math.max(...r.results.map(x => x["R@1"]));
    const rows = r.results.map(x => `<tr><td>${MODE_LABEL[x.mode]}</td>${["R@1", "R@5", "R@10", "MRR@10", "nDCG@10"].map(k =>
      `<td class="num">${x[k] === best && k === "R@1" ? `<b>${x[k]}</b>` : x[k]}</td>`).join("")}<td class="num">${x.elapsed_s}s</td></tr>`).join("");
    const worst = r.results.find(x => x.mode === "hybrid") || r.results[0];
    const fails = worst.failures.slice(0, 10).map(f => `<tr><td>${esc(f.query)}</td><td class="num">${f.gold_rank ?? "未进前 10"}</td>
      <td>${f.gold.map(g => `<a href="#" data-open="${esc(g)}">${esc(g)}</a>`).join(" ")}</td></tr>`).join("");
    $("#ev-res").innerHTML = `<h3>${esc(name)} · ${r.n} 条 · ${$("#ev-dim").value} 维</h3>
      <div class="scroll"><table><thead><tr><th>召回方式</th><th class="num">R@1</th><th class="num">R@5</th><th class="num">R@10</th><th class="num">MRR@10</th><th class="num">nDCG@10</th><th class="num">用时</th></tr></thead><tbody>${rows}</tbody></table></div>
      <h3>错例（${MODE_LABEL[worst.mode]}，标准答案不在第 1 名）</h3>
      <div class="scroll"><table class="fail"><thead><tr><th>查询</th><th class="num">答案名次</th><th>标准答案</th></tr></thead><tbody>${fails || '<tr><td colspan="3" class="mute">全部第 1 名命中</td></tr>'}</tbody></table></div>`;
    $("#ev-res").querySelectorAll("[data-open]").forEach(a => a.onclick = e => { e.preventDefault(); openDetail(a.dataset.open); });
  } catch (e) { $("#ev-res").innerHTML = `<p class="err">运行失败：${esc(e.message)}</p>`; }
  btn.disabled = false;
};

// 打开时按地址栏 #标签 定位
{
  const h = (location.hash || "").slice(1);
  showTab(["ingest", "jobs", "library", "search", "eval", "health"].includes(h) ? h : "ingest");
}
