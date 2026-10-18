"use strict";

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const number = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
  const count = (value) => number(value).toLocaleString("zh-CN");
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const icon = (name, cls = "") => `<svg${cls ? ` class="${escape(cls)}"` : ""} aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const modeNames = { hybrid: "混合检索", vector: "语义检索", bm25: "关键词检索" };
  const actionNames = { sync: "同步资料库", scan: "扫描与建立索引", embed: "补齐向量" };
  const statusNames = { blocked: "未通过门槛", accepted: "通过本轮门槛", superseded: "已被替代", encoded: "子集编码完成", built: "子集已构建", queued: "等待执行", pending: "等待执行", running: "进行中", scanning: "扫描中", embedding: "向量化中", cancelling: "正在停止", canceling: "正在停止", completed: "已完成", complete: "已完成", done: "已完成", succeeded: "已完成", partial: "部分完成", paused: "已暂停", failed: "失败", error: "失败", cancelled: "已停止", canceled: "已停止", interrupted: "已中断" };
  const phaseNames = { scan: "扫描目录", sync: "扫描与编码交错执行", embedding_interleaved: "扫描与编码交错执行", scanning: "扫描目录", parse: "解析文件", parsing: "解析文件", chunk: "建立片段", chunking: "建立片段", index: "建立索引", indexing: "建立索引", embed: "向量化", embedding: "向量化", vectorizing: "向量化", complete: "处理完成", completed: "处理完成", done: "处理完成", queued: "等待开始", pending: "等待开始" };
  const doneStatuses = new Set(["completed", "complete", "done", "succeeded", "partial", "paused", "failed", "error", "cancelled", "canceled", "interrupted"]);
  const state = { view: "search", mode: "hybrid", project: "", overview: null, health: null, jobs: null, search: null, results: new Map(), traces: new Map(), jobRecords: new Map(), batches: [], scanSummary: {}, selectedEvaluation: "", evaluationSerial: 0, searchController: null, searchSerial: 0, librarySerial: 0, libraryOffset: 0, libraryTotal: 0, exclusionSerial: 0, exclusionOffset: 0, exclusionTotal: 0, detailSerial: 0, detailData: null, toastTimer: null, pollTimer: null, submitting: false, polling: false };
  const detail = $("#detail-dialog");

  async function api(path, options = {}) {
    const response = await fetch(`/api/code${path}`, { credentials: "same-origin", ...options, headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers } });
    const text = await response.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch { throw new Error(`服务返回了无法读取的内容（HTTP ${response.status}）。`); }
    if (!response.ok) {
      const message = typeof data.detail === "string" ? data.detail : data.error?.message || data.error || data.message;
      throw new Error(typeof message === "string" ? message : `请求失败（HTTP ${response.status}）。`);
    }
    return data;
  }

  function toast(message) {
    $("#toast").textContent = message;
    $("#toast").hidden = false;
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 4300);
  }

  function globalError(message) {
    $("#global-notice").textContent = message;
    $("#global-notice").hidden = !message;
  }

  function formatBytes(value) {
    const bytes = number(value);
    if (bytes < 1024) return `${count(bytes)} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  }

  function score(value, digits = 5) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return Number(value).toFixed(digits);
  }

  function time(value) {
    if (!value) return "—";
    let normalized = value;
    if (typeof value === "number") normalized = value < 1e12 ? value * 1000 : value;
    if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(value) && !/[zZ]|[+-]\d{2}:?\d{2}$/.test(value)) normalized = value.replace(" ", "T") + "Z";
    const date = new Date(normalized);
    return Number.isNaN(date.getTime()) ? String(value) : new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(date);
  }

  function active(job) { return !!job && !doneStatuses.has(job.status); }
  function jobClass(job) { return ["failed", "error"].includes(job.status) ? "failed" : ["partial", "paused", "interrupted"].includes(job.status) ? "partial" : active(job) ? "running" : ["completed", "complete", "done", "succeeded"].includes(job.status) ? "completed" : ""; }
  function jobStatus(job) { return `<span class="status-pill ${jobClass(job)}">${escape(statusNames[job.status] || job.status || "未知状态")}</span>`; }
  function basename(path) { return String(path || "").split(/[\\/]/).pop() || "未命名文件"; }
  function dirname(path) { const parts = String(path || "").split(/[\\/]/); parts.pop(); return parts.join("/") || "."; }
  function ranks(rank) { return rank === null || rank === undefined ? "未入选" : `#${count(rank)}`; }

  function object(value) { if (value && typeof value === "object" && !Array.isArray(value)) return value; if (typeof value === "string") { try { const parsed = JSON.parse(value); return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {}; } catch { /* Older servers may return an empty or non-JSON report. */ } } return {}; }
  function percent(completed, total) { const done = Math.max(0, number(completed)), all = Math.max(0, number(total)); if (!all) return "0"; if (done >= all) return "100"; const result = Math.floor(done / all * 1000) / 10; return result === 0 && done > 0 ? "<0.1" : result.toFixed(1).replace(/\.0$/, ""); }
  function selectValue(selector, value, label) { const select = $(selector), requested = String(value ?? ""); if (requested && ![...select.options].some((option) => option.value === requested)) { const option = document.createElement("option"); option.value = requested; option.textContent = label || requested; select.appendChild(option); } select.value = requested; }
  function restoreSearch(params) {
    setMode(modeNames[params.mode] ? params.mode : "hybrid");
    $("#query").value = String(params.query || "").slice(0, 4000);
    const project = params.project || "", extension = params.extension || "";
    selectValue("#project-filter", project); setProject(project);
    selectValue("#extension-filter", extension);
    const limit = Math.min(100, Math.max(1, Math.floor(number(params.limit) || 10)));
    const candidates = Math.min(1000, Math.max(limit, Math.floor(number(params.candidates) || 100)));
    selectValue("#result-limit", limit, `${limit} 条`); selectValue("#candidates", candidates);
    const weight = params.vector_weight !== undefined && params.vector_weight !== null && Number.isFinite(Number(params.vector_weight)) ? Math.min(1, Math.max(0, Number(params.vector_weight))) : .6;
    $("#vector-weight").step = "any"; $("#vector-weight").value = String(weight * 100); $("#weight-value").textContent = `${Number((weight * 100).toFixed(3))}%`;
    syncSearchDraft();
  }

  function writeSearchURL(params) { const url = new URL(location.href); for (const key of ["q", "query", "mode", "project", "extension", "limit", "candidates", "vector_weight"]) url.searchParams.delete(key); url.searchParams.set("q", params.query); url.searchParams.set("mode", params.mode); for (const key of ["project", "extension", "limit", "candidates", "vector_weight"]) if (params[key] !== null && params[key] !== undefined && params[key] !== "") url.searchParams.set(key, String(params[key])); history.replaceState(null, "", url); }

  function searchParameters() {
    return { query: $("#query").value.trim(), mode: state.mode, limit: number($("#result-limit").value), candidates: Math.max(number($("#candidates").value), number($("#result-limit").value)), vector_weight: number($("#vector-weight").value) / 100, project: $("#project-filter").value || null, extension: $("#extension-filter").value || null };
  }

  function syncSearchDraft() {
    const draft = searchParameters(), submitted = state.pendingRequest || state.search?.request;
    const changed = !!submitted && Object.keys(draft).some((key) => draft[key] !== submitted[key]);
    $("#search-draft").hidden = !changed;
    $("#search-draft > span").textContent = state.pendingRequest ? "正在按已提交的条件搜索；新修改尚未生效。" : "检索条件已修改，当前仍显示上次结果。";
    $("#view-search").classList.toggle("draft-changed", changed);
    const advanced = [!!draft.extension, draft.limit !== 10, draft.candidates !== 100, draft.mode === "hybrid" && draft.vector_weight !== .6].filter(Boolean).length;
    $("#advanced-count").textContent = String(advanced);
    $("#advanced-count").hidden = !advanced;
  }

  function renderSearchWarnings() {
    const data = state.search, warnings = [];
    if (data) {
      if (data.warning) warnings.push(data.warning);
      const coverage = data.explanation?.coverage;
      if (coverage && (data.warning || data.indexChanged || number(coverage.vectors) < number(coverage.chunks))) warnings.push(`查询时的向量覆盖：${count(coverage.vectors)} / ${count(coverage.chunks)} 个片段${data.request.project ? `（${data.request.project}）` : ""}。`);
      if (data.indexChanged && !warnings.length) warnings.push("本次结果尚未重新检索。");
      if (warnings.length) $("#search-warning").textContent = `${data.indexChanged ? "资料库数量已变化，可重新搜索。以下保留 " : ""}${time(data.searchedAt)} 检索时的记录：${warnings.join(" ")}`;
    }
    // Keep the response's coverage and warning together. A fresh overview is
    // not evidence that the already displayed results used the fresh index.
    $("#search-warning").hidden = !warnings.length;
  }

  function resultContext(params) {
    return `「${params.query}」 · ${params.project || "全部项目"}${params.extension ? ` · ${params.extension}` : ""} · ${modeNames[params.mode]}`;
  }

  function highlight(content, terms) {
    const source = String(content ?? "");
    const words = [...new Set((Array.isArray(terms) ? terms : []).filter((t) => typeof t === "string" && t.length && t.length < 160))].slice(0, 40).sort((a, b) => b.length - a.length);
    if (!words.length) return escape(source);
    const re = new RegExp(words.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|"), "gi");
    let html = "", index = 0;
    for (const match of source.matchAll(re)) {
      html += escape(source.slice(index, match.index)) + `<mark>${escape(match[0])}</mark>`;
      index = match.index + match[0].length;
    }
    return html + escape(source.slice(index));
  }

  function setView(view) {
    if (!["search", "library", "jobs", "exclusions", "evaluations"].includes(view)) return;
    state.view = view;
    $$(".view").forEach((el) => { el.hidden = el.id !== `view-${view}`; });
    $$(".nav-item").forEach((el) => { const selected = el.dataset.view === view; el.classList.toggle("active", selected); selected ? el.setAttribute("aria-current", "page") : el.removeAttribute("aria-current"); });
    $("#breadcrumb-current").textContent = { search: "探索检索", library: "资料库", jobs: "索引任务", exclusions: "跳过清单", evaluations: "质量评测" }[view];
    $("#sync-button").hidden = !["library", "jobs"].includes(view);
    document.title = `${{ search: "探索检索", library: "资料库", jobs: "索引任务", exclusions: "跳过清单", evaluations: "质量评测" }[view]} · OmniIndex`;
    closeSidebar();
    if (view === "library") loadLibrary();
    if (view === "jobs") Promise.allSettled([loadJobs().then(() => loadBatches()), loadTraces()]);
    if (view === "exclusions") { loadExclusions(); renderDirectorySamples(); }
    if (view === "evaluations") loadEvaluations();
  }

  function setMode(mode) {
    if (!modeNames[mode]) return;
    state.mode = mode;
    $$(".mode-button").forEach((el) => { const selected = el.dataset.mode === mode; el.classList.toggle("selected", selected); el.setAttribute("aria-checked", String(selected)); el.tabIndex = selected ? 0 : -1; });
    $("#algorithm-label").textContent = modeNames[mode];
    $("#mode-description").textContent = { hybrid: "结合语义与关键词，按两路排名融合。向量覆盖不足时自动使用关键词；修改设置后重新搜索生效。", vector: "只搜索已有向量的内容。大资料库可先选择项目，缩小检索范围。", bm25: "使用本地 BM25 召回与词项重排，不调用查询编码。可在结果详情查看原始分数与重排依据。" }[mode];
    $("#vector-weight").disabled = mode !== "hybrid";
    syncSearchDraft();
  }

  function setProject(project, { navigate = false } = {}) {
    state.project = project || "";
    selectValue("#project-filter", state.project);
    $$(".project-item").forEach((el) => { const selected = el.dataset.project === state.project; el.classList.toggle("active", selected); el.setAttribute("aria-pressed", String(selected)); });
    if (navigate) setView("search");
    syncSearchDraft();
  }

  function renderOverview(data) {
    state.overview = data;
    if (state.search?.overviewAtSearch && ["files", "chunks", "vectors"].some((key) => number(data[key]) !== state.search.overviewAtSearch[key])) state.search.indexChanged = true;
    const files = number(data.files), chunks = number(data.chunks), vectors = number(data.vectors);
    const projects = Array.isArray(data.projects) ? data.projects : [];
    $("#metric-files").textContent = count(files);
    $("#metric-chunks").textContent = count(chunks);
    $("#metric-coverage").innerHTML = `${percent(vectors, chunks)}<small>%</small>`;
    $("#metric-dimensions").textContent = data.dimensions ? count(data.dimensions) : "—";
    $("#metric-projects").textContent = `${count(projects.length)} 个项目目录`;
    $("#metric-pending").textContent = chunks ? `${count(vectors)} / ${count(chunks)} 个片段 · ${count(Math.max(0, chunks - vectors))} 待向量化` : "0 / 0 个片段 · 等待扫描目录";
    $("#metric-model").textContent = data.model || "尚未配置向量模型";
    $("#metric-model").title = [data.model, data.revision].filter(Boolean).join(" @ ");
    $("#nav-file-count").textContent = count(files);
    $("#project-count").textContent = count(projects.length);
    $("#storage-path").textContent = data.root || "未配置来源目录";
    $("#storage-path").title = data.root || "";
    $("#library-root").textContent = data.root || "未配置来源目录";
    $("#storage-size").textContent = formatBytes(data.storage_bytes);
    $("#project-list").innerHTML = `<button type="button" class="project-item${!state.project ? " active" : ""}" data-project="" aria-pressed="${!state.project}">${icon("layers")}<span>全部项目</span><small>${count(files)}</small></button>` + projects.map((p) => `<button type="button" class="project-item${state.project === p.name ? " active" : ""}" data-project="${escape(p.name)}" aria-pressed="${state.project === p.name}" title="${escape(p.name)} · ${count(p.chunks)} 个片段">${icon("folder")}<span>${escape(p.name || "根目录")}</span><small>${count(p.files)}</small></button>`).join("");
    for (const target of ["#project-filter", "#library-project"]) {
      const select = $(target), previous = select.value;
      select.innerHTML = `<option value="">全部项目</option>` + projects.map((p) => `<option value="${escape(p.name)}">${escape(p.name || "根目录")} (${count(p.files)})</option>`).join("");
      selectValue(target, previous);
    }
    const extensionSelect = $("#extension-filter"), previousExtension = extensionSelect.value;
    const extensions = Array.isArray(data.extensions) ? data.extensions : [];
    extensionSelect.innerHTML = `<option value="">全部类型</option>` + extensions.map((e) => `<option value="${escape(e.extension)}">${escape(e.extension || "无扩展名")} (${count(e.files)})</option>`).join("");
    selectValue("#extension-filter", previousExtension);
    const exclusions = Array.isArray(data.exclusions) ? data.exclusions : [];
    $("#exclusions").innerHTML = exclusions.length ? exclusions.map((e) => `<span>${escape(typeof e === "string" ? e : JSON.stringify(e))}</span>`).join("") : `<span>暂无额外排除配置</span>`;
    state.scanSummary = object(data.scan_summary);
    renderScanSummary(); renderDirectorySamples();
    renderJobBanner(data.job);
    updateActionButtons();
    renderSearchWarnings();
  }

  async function loadOverview() { const data = await api("/overview"); renderOverview(data); return data; }

  function renderHealth(health) {
    state.health = health;
    $("#service-dot").className = `status-dot ${health.ready ? "green" : health.configured ? "amber" : "red"}`;
    $("#service-label").textContent = health.ready ? "推理服务已就绪" : health.configured ? "推理服务待就绪" : "推理服务未配置";
    $("#health-button").title = health.message || (health.ready ? "推理服务可用；点击刷新状态" : "点击刷新状态");
  }

  async function loadHealth() {
    try { renderHealth(await api("/health")); }
    catch (error) { $("#service-dot").className = "status-dot amber"; $("#service-label").textContent = "服务状态暂不可读"; $("#health-button").title = error.message; throw error; }
  }

  function renderJobBanner(job) {
    const running = active(job);
    $("#index-progress").hidden = !running;
    $("#nav-job-indicator").hidden = !running;
    if (running) {
      $("#index-progress-title").textContent = `${actionNames[job.action] || "索引任务"}进行中`;
      const progress = jobProgress(job);
      $("#index-progress-description").textContent = `${progress.scanOpen ? `已发现 ${count(job.files_done)} 个文件 · 遍历未完成` : phaseNames[progress.lastPhase] || job.phase || "处理中"}${progress.hasVectors ? ` · 本任务已保存 ${count(job.vectors_done)} 个向量${progress.scanOpen ? "（仍会发现新片段）" : ` / ${count(job.vectors_total)}`}` : " · 尚无已完成编码记录"}`;
    }
  }

  async function runSearch() {
    const query = $("#query").value.trim();
    if (!query) { $("#query").focus(); return; }
    if (state.searchController) state.searchController.abort();
    state.searchController = new AbortController();
    const serial = ++state.searchSerial;
    const payload = searchParameters();
    state.pendingRequest = payload;
    state.search = null;
    $("#view-search").classList.add("has-search");
    $("#search-results-heading").hidden = false;
    $("#results-context").textContent = resultContext(payload);
    $("#results-context").title = resultContext(payload);
    writeSearchURL(payload);
    syncSearchDraft();
    $("#search-submit").disabled = true;
    $("#search-submit").innerHTML = `<span class="spinner" aria-hidden="true"></span><span>检索中</span>`;
    $("#results").setAttribute("aria-busy", "true");
    $("#results").innerHTML = `<div class="results-loading"><span class="spinner" aria-hidden="true"></span>正在检索本地资料库…</div>`;
    $("#results-title").textContent = "正在查找";
    $("#results-meta").textContent = `以${modeNames[payload.mode]}查询${payload.project ? `「${payload.project}」` : "全部项目"}`;
    $("#search-warning").hidden = true;
    $("#show-trace").hidden = true;
    state.results.clear();
    try {
      const data = await api("/search", { method: "POST", body: JSON.stringify(payload), signal: state.searchController.signal });
      if (serial !== state.searchSerial) return;
      state.search = { ...data, request: payload, searchedAt: new Date().toISOString(), overviewAtSearch: Object.fromEntries(["files", "chunks", "vectors"].map((key) => [key, number(state.overview?.[key])])) };
      const results = Array.isArray(data.results) ? data.results : [];
      results.forEach((r) => state.results.set(String(r.id), r));
      $("#results-title").textContent = results.length ? "检索结果" : "暂无匹配结果";
      const elapsedMs = typeof data.elapsed_ms === "number" ? data.elapsed_ms : NaN;
      const elapsed = Number.isFinite(elapsedMs) ? `${(elapsedMs / 1000).toFixed(elapsedMs < 1000 ? 2 : 1)} 秒` : "耗时未知";
      $("#results-meta").textContent = `${count(results.length)} 条结果 · ${modeNames[data.effective_mode || data.mode] || data.effective_mode || modeNames[payload.mode]} · ${elapsed}`;
      $("#show-trace").hidden = false;
      renderSearchWarnings();
      $("#results").innerHTML = results.length ? results.map(renderResult).join("") : `<div class="plain-empty"><strong>换个说法，再找一次</strong>可以减少限定词、切换召回方式，或检查资料库的索引进度。<div class="empty-actions"><button class="button small" type="button" data-open-library>查看资料库</button><button class="button small" type="button" data-open-jobs>查看索引任务</button></div></div>`;
    } catch (error) {
      if (error.name === "AbortError" || serial !== state.searchSerial) return;
      $("#results-title").textContent = "检索未完成";
      $("#results-meta").textContent = "请检查服务连接或稍后重试";
      $("#results").innerHTML = `<div class="plain-empty"><strong>暂时无法完成检索</strong>${escape(error.message)}<div class="empty-actions"><button type="button" class="button small" data-retry-search>重新检索</button><button type="button" class="button small" data-open-jobs>查看任务与服务</button></div></div>`;
    } finally {
      if (serial === state.searchSerial) {
        state.pendingRequest = null;
        syncSearchDraft();
        $("#search-submit").disabled = false;
        $("#search-submit").innerHTML = `<span>搜索</span>${icon("arrow")}`;
        $("#results").setAttribute("aria-busy", "false");
      }
    }
  }

  function renderResult(result, index) {
    const snippet = String(result.content || "");
    const preview = snippet.length > 1600 ? snippet.slice(0, 1600) + "\n… 点击查看完整片段" : snippet;
    const label = (state.search?.effective_mode || state.search?.mode) === "vector" ? "相似度" : (state.search?.effective_mode || state.search?.mode) === "bm25" ? "关键词分" : "融合分";
    const vectorTag = result.vector_rank !== null && result.vector_rank !== undefined ? `<span class="route-tag vector">语义 ${score(result.vector_score, 4)}<span class="route-rank">#${count(result.vector_rank)}</span></span>` : "";
    const bm25Tag = result.bm25_rank !== null && result.bm25_rank !== undefined ? `<span class="route-tag bm25">关键词 ${score(result.bm25_score, 3)}<span class="route-rank">#${count(result.bm25_rank)}</span></span>` : "";
    return `<article class="result-card"><button class="result-header" type="button" data-chunk="${escape(result.id)}" aria-label="查看 ${escape(result.path)} 第 ${count(result.start_line)} 至 ${count(result.end_line)} 行"><span class="result-order">${String(index + 1).padStart(2, "0")}</span><span class="file-type-icon">${icon("file")}</span><span class="result-name-block"><span class="result-filename">${escape(basename(result.path))}</span><span class="result-path">${escape(dirname(result.path))} <span>· L${count(result.start_line)}–${count(result.end_line)}</span></span></span><span class="result-score">${label}<span class="score-badge">${score(result.score, label === "融合分" ? 5 : 4)}</span></span>${icon("arrow")}</button><div class="result-snippet"><pre>${highlight(preview, result.matched_terms)}</pre></div><div class="result-footer"><div class="result-route-tags">${vectorTag}${bm25Tag}${!vectorTag && !bm25Tag ? `<span class="route-tag">${escape(result.project || "根目录")}</span>` : ""}</div><button class="text-button" type="button" data-chunk="${escape(result.id)}">查看原文与依据 ${icon("external")}</button></div></article>`;
  }

  async function loadLibrary() {
    const serial = ++state.librarySerial;
    $("#file-rows").innerHTML = `<tr><td colspan="6" class="table-empty"><span class="spinner" aria-hidden="true"></span> 正在读取文件…</td></tr>`;
    const params = new URLSearchParams({ limit: "50", offset: String(state.libraryOffset) });
    if ($("#library-project").value) params.set("project", $("#library-project").value);
    if ($("#library-query").value.trim()) params.set("q", $("#library-query").value.trim());
    try {
      const data = await api(`/files?${params}`);
      if (serial !== state.librarySerial) return;
      const files = Array.isArray(data.items) ? data.items : [];
      state.libraryTotal = number(data.total);
      $("#library-count").textContent = `${count(data.total)} 个文件`;
      $("#file-rows").innerHTML = files.length ? files.map(renderFile).join("") : `<tr><td colspan="6" class="table-empty">${params.has("q") || params.has("project") ? "没有找到符合条件的文件。" : "资料库还没有文件。点击「扫描目录」建立本地索引。"}</td></tr>`;
      $("#library-page-label").textContent = data.total ? `第 ${count(state.libraryOffset + 1)}–${count(Math.min(state.libraryOffset + files.length, data.total))} 项，共 ${count(data.total)} 项` : "暂无文件";
      $("#library-prev").disabled = state.libraryOffset === 0;
      $("#library-next").disabled = state.libraryOffset + 50 >= state.libraryTotal;
    } catch (error) {
      if (serial !== state.librarySerial) return;
      $("#file-rows").innerHTML = `<tr><td colspan="6" class="table-empty">${escape(error.message)}</td></tr>`;
      $("#library-count").textContent = "读取失败";
      $("#library-page-label").textContent = "文件列表暂不可用";
      $("#library-prev").disabled = true; $("#library-next").disabled = true;
    }
  }

  function renderFile(file) {
    const chunks = number(file.chunk_count), vectors = number(file.vector_count);
    const ready = chunks > 0 && vectors >= chunks, error = !!file.error;
    return `<tr><td><button class="file-button" type="button" data-file="${escape(file.id)}">${icon("file")}<span><strong>${escape(basename(file.path))}</strong><small>${escape(dirname(file.path))}</small></span></button></td><td>${escape(file.project || "根目录")}</td><td class="numeric">${formatBytes(file.size)}</td><td class="numeric">${count(chunks)}</td><td><span class="file-status" title="${escape(file.error || file.status || "")}"><span class="status-dot ${error ? "red" : ready ? "green" : "amber"}"></span>${error ? "处理异常" : ready ? "已就绪" : chunks ? "待补齐" : "无片段"}<small>${count(vectors)}/${count(chunks)}</small></span></td><td><button type="button" class="icon-button" data-file="${escape(file.id)}" aria-label="查看 ${escape(file.path)}">${icon("chevron")}</button></td></tr>`;
  }

  function jobProgress(job) {
    const details = object(job.details);
    const lastPhase = details.last_work_phase || details.last_phase || (number(job.vectors_total) > 0 || number(job.vectors_done) > 0 ? "embed" : job.phase);
    const interleaved = ["sync", "embedding_interleaved"].includes(job.phase) || !!details.embedding_interleaved || !!details.interleaved;
    const declaredScanDone = details.scan_complete === true || details.scan_completed === true || details.scan_finished === true;
    const declaredScanPending = details.scan_complete === false || details.scan_completed === false || details.scan_finished === false;
    const scanComplete = job.action === "embed" || (!declaredScanPending && (declaredScanDone || (!interleaved && active(job) && ["embed", "embedding", "vectorizing"].includes(job.phase)) || ["completed", "complete", "done", "succeeded", "partial"].includes(job.status)));
    const scanOpen = job.action !== "embed" && active(job) && !scanComplete;
    const hasVectors = job.action === "embed" || number(job.vectors_total) > 0 || number(job.vectors_done) > 0 || ["embed", "embedding", "vectorizing"].includes(lastPhase);
    return {details, lastPhase, interleaved, scanComplete, scanOpen, hasVectors, vectorKnown: number(job.vectors_total) > 0 && !scanOpen && (scanComplete || !interleaved)};
  }

  async function loadJobs() {
    const data = await api("/jobs");
    state.jobs = data;
    for (const job of [data.job, ...(Array.isArray(data.recent) ? data.recent : [])]) if (job) state.jobRecords.set(String(job.id), job);
    if (state.overview) state.overview.job = data.job;
    renderJobBanner(data.job); updateActionButtons();
    $("#current-job").innerHTML = data.job ? renderCurrentJob(data.job) : `<div class="plain-empty"><strong>准备好，把内容连接起来</strong>点击「同步资料库」，扫描来源目录并生成向量。<div class="empty-actions"><button type="button" class="button small primary" data-start-job="sync">${icon("refresh")}同步资料库</button></div></div>`;
    const recent = (Array.isArray(data.recent) ? data.recent : []).filter((job) => !data.job || String(job.id) !== String(data.job.id));
    $("#recent-job-count").textContent = `${count(recent.length)} 条记录`;
    $("#recent-jobs").innerHTML = recent.length ? recent.map((job) => `<div class="recent-row">${icon(job.action === "embed" ? "spark" : "folder")}<div class="recent-row-main"><strong>${escape(actionNames[job.action] || job.action || "索引任务")}</strong><small>文件 ${count(job.files_done)} · 片段 ${count(job.chunks_done)} · 向量 ${count(job.vectors_done)}${job.errors ? ` · 异常 ${count(Array.isArray(job.errors) ? job.errors.length : job.errors)}` : ""}</small></div><span class="recent-row-time">${escape(time(job.started_at))}</span>${jobStatus(job)}<button type="button" class="text-button" data-job="${escape(job.id)}">详情 ${icon("chevron")}</button></div>`).join("") : `<div class="plain-empty">暂无历史任务</div>`;
    renderDirectorySamples();
    return data;
  }

  function renderCurrentJob(job, inspector = false) {
    const progress = jobProgress(job), running = active(job), details = progress.details;
    const errorCount = Array.isArray(job.errors) ? job.errors.length : number(job.errors);
    const message = typeof job.message === "string" ? job.message : job.message ? JSON.stringify(job.message) : "";
    const vectorPct = Math.min(100, Math.max(0, number(job.vectors_done) / Math.max(1, number(job.vectors_total)) * 100));
    const scanBlock = job.action !== "embed" ? `<div class="job-progress-unit"><div class="job-progress-labels"><strong>目录扫描</strong><span>${progress.scanComplete ? `扫描已结束 · ${count(job.files_done)} 个文件` : `已发现 ${count(job.files_done)} 个文件 · ${progress.scanOpen ? "持续遍历，总数未知" : "扫描未完成，总数未知"}`}</span></div><div class="progress-track${progress.scanOpen ? " indeterminate" : ""}" aria-label="目录扫描状态"><span style="width:${progress.scanComplete ? 100 : progress.scanOpen ? 30 : 0}%"></span></div></div>` : "";
    const vectorBlock = job.action !== "scan" ? `<div class="job-progress-unit"><div class="job-progress-labels"><strong>本任务向量编码</strong><span>${progress.vectorKnown ? `${count(job.vectors_done)} / ${count(job.vectors_total)} · ${percent(job.vectors_done, job.vectors_total)}%` : progress.hasVectors ? `已保存 ${count(job.vectors_done)} 个 · ${progress.scanOpen ? "扫描中，待处理总量仍会变化" : "待处理总量未知"}` : "尚无已完成编码记录"}</span></div><div class="progress-track${running && !progress.vectorKnown && progress.hasVectors ? " indeterminate" : ""}" role="progressbar" aria-label="本任务已知片段编码进度" aria-valuemin="0" aria-valuemax="100"${progress.vectorKnown ? ` aria-valuenow="${Math.floor(vectorPct * 10) / 10}"` : ""}><span style="width:${progress.vectorKnown ? vectorPct : 0}%"></span></div></div>` : "";
    const recoveryAction = job.action === "scan" ? "scan" : progress.scanComplete || job.action === "embed" ? "embed" : "sync";
    const recoveryText = recoveryAction === "embed" ? "继续补齐向量" : recoveryAction === "scan" ? "重新扫描" : "继续同步";
    const controls = inspector ? "" : running ? `<button class="button small danger" type="button" data-cancel-job ${["cancelling", "canceling"].includes(job.status) ? "disabled" : ""}>停止任务</button>` : `<button class="button small" type="button" data-start-job="${recoveryAction}">${recoveryText}</button>`;
    const lastBatch = object(details.last_batch);
    return `<div class="job-top"><span class="job-icon">${icon(job.action === "embed" ? "spark" : "activity")}</span><div class="job-headline"><h3>${escape(actionNames[job.action] || job.action || "索引任务")}${jobStatus(job)}</h3><p>开始于 ${escape(time(job.started_at))}${job.finished_at ? ` · 结束于 ${escape(time(job.finished_at))}` : ""} · 北京时间</p></div>${controls}${!inspector ? `<button class="text-button" type="button" data-job="${escape(job.id)}">任务详情 ${icon("chevron")}</button>` : ""}</div><div class="job-progress-block">${scanBlock}${vectorBlock}<p class="jobs-note">${progress.scanOpen ? "扫描与编码可交错执行；已发现文件数不能作为整个任务的完成比例。" : "以上分别表示扫描与本任务编码进度；全库覆盖率可在资料库查看；与本任务进度分开统计。"}${progress.lastPhase ? ` 最近工作阶段：${escape(phaseNames[progress.lastPhase] || progress.lastPhase)}。` : ""}</p></div><div class="job-stats"><div>已发现文件<strong>${count(job.files_done)}</strong></div><div>已建立片段<strong>${count(job.chunks_done)}</strong></div><div>本任务已保存向量<strong>${count(job.vectors_done)}</strong></div><div>跳过 / 异常<strong>${count(job.skipped)} <span class="muted">/ ${count(errorCount)}</span></strong></div></div>${message ? `<div class="job-message${["failed", "error"].includes(job.status) ? " error" : ""}">${escape(message)}</div>` : ""}${Object.keys(lastBatch).length ? `<div class="job-message">最近编码批次：${escape(lastBatch.status || "已记录")} · ${lastBatch.elapsed_ms != null ? `${number(lastBatch.elapsed_ms).toFixed(0)} ms` : "耗时待记录"} · ${count(Array.isArray(lastBatch.items) ? lastBatch.items.length : lastBatch.items)} 个片段${lastBatch.http_status ? ` · HTTP ${escape(lastBatch.http_status)}` : ""}</div>` : ""}${Array.isArray(job.errors) && job.errors.length ? `<div class="job-message error">${job.errors.slice(0, 8).map((e) => escape(typeof e === "string" ? e : JSON.stringify(e))).join("<br>")}</div>` : ""}${Object.keys(details).length ? `<details class="raw-json job-raw-details"${!inspector && $("#current-job .job-raw-details")?.open ? " open" : ""}><summary>完整任务观测与扫描原因</summary><pre>${escape(JSON.stringify(details, null, 2))}</pre></details>` : ""}`;
  }

  function renderScanSummary() {
    const report = state.scanSummary;
    if (!Object.keys(report).length) { $("#scan-report").innerHTML = `<div class="plain-empty">尚无已结束的扫描报告。进行中的任务可在上方查看。</div>`; return; }
    $("#scan-report").innerHTML = `<div class="job-stats"><div>登记文件<strong>${count(report.files)}</strong></div><div>建立片段<strong>${count(report.chunks)}</strong></div><div>跳过文件 / 异常<strong>${count(report.skipped)} / ${count(report.errors)}</strong></div><div>排除目录<strong>${count(report.excluded_directories)}</strong></div></div><div class="job-message">最近扫描：${escape(time(report.scanned_at))}（北京时间）${report.prune_deferred ? " · 存在目录读取异常，本轮未删除未发现的旧记录。" : ""}<button class="text-button" type="button" data-open-exclusions> 查看跳过清单 ↗</button></div><details class="raw-json job-raw-details"><summary>持久化扫描报告与原因统计</summary><pre>${escape(JSON.stringify(report, null, 2))}</pre></details>`;
  }

  function directorySamples() {
    const latest = object(state.jobs?.job?.details), persisted = state.scanSummary;
    const samples = [...(Array.isArray(latest.skip_samples) ? latest.skip_samples : []), ...(Array.isArray(persisted.skip_samples) ? persisted.skip_samples : [])];
    const seen = new Set();
    return samples.filter((item) => { if (!item?.directory || seen.has(item.path)) return false; seen.add(item.path); return true; });
  }

  function renderDirectorySamples() {
    const samples = directorySamples();
    $("#excluded-directories").innerHTML = samples.length ? `<div class="directory-sample-note">${count(samples.length)} 个目录样例（数量受任务记录采样上限约束）</div>` + samples.map((sample) => `<div class="directory-sample">${icon("folder")}<code>${escape(sample.path)}</code><span>${escape(sample.reason || "按扫描规则排除")}</span></div>`).join("") : `<div class="plain-empty">当前任务与最近扫描报告中没有保留目录排除样例。</div>`;
  }

  async function loadExclusions() {
    const serial = ++state.exclusionSerial;
    $("#exclusion-rows").innerHTML = `<tr><td colspan="5" class="table-empty"><span class="spinner" aria-hidden="true"></span> 正在读取已登记的跳过文件…</td></tr>`;
    const params = new URLSearchParams({ limit: "50", offset: String(state.exclusionOffset) });
    for (const [key, selector] of [["q", "#exclusion-query"], ["reason", "#exclusion-reason"], ["project", "#exclusion-project"]]) if ($(selector).value.trim()) params.set(key, $(selector).value.trim());
    try {
      const data = await api(`/exclusions?${params}`);
      if (serial !== state.exclusionSerial) return;
      const items = Array.isArray(data.items) ? data.items : [];
      state.exclusionTotal = number(data.total);
      $("#exclusion-count").textContent = `${count(data.total)} 个已登记文件`;
      $("#exclusion-rows").innerHTML = items.length ? items.map((file) => `<tr><td><span class="exclusion-path">${escape(basename(file.path))}<small>${escape(dirname(file.path))}</small></span></td><td>${escape(file.project || "根目录")}</td><td class="numeric">${formatBytes(file.size)}</td><td><span class="status-pill${file.status === "error" ? " failed" : ""}">${escape(file.status === "error" ? "读取异常" : file.status === "skipped" ? "已跳过" : file.status || "未收录")}</span></td><td class="exclusion-reason-cell">${escape(file.error || file.reason || "按配置排除")}</td></tr>`).join("") : `<tr><td colspan="5" class="table-empty">没有符合当前条件的已登记跳过文件。整目录排除请查看下方样例。</td></tr>`;
      const selected = $("#exclusion-reason").value;
      $("#exclusion-reason").innerHTML = `<option value="">全部原因</option>` + (Array.isArray(data.reasons) ? data.reasons : []).map((item) => `<option value="${escape(item.reason)}">${escape(item.reason)} (${count(item.count)})</option>`).join("");
      selectValue("#exclusion-reason", selected);
      $("#exclusion-page-label").textContent = data.total ? `第 ${count(state.exclusionOffset + 1)}–${count(Math.min(state.exclusionOffset + items.length, data.total))} 项，共 ${count(data.total)} 项` : "暂无匹配文件";
      $("#exclusion-prev").disabled = !state.exclusionOffset;
      $("#exclusion-next").disabled = state.exclusionOffset + 50 >= state.exclusionTotal;
      renderDirectorySamples();
    } catch (error) {
      if (serial !== state.exclusionSerial) return;
      $("#exclusion-rows").innerHTML = `<tr><td colspan="5" class="table-empty">无法读取逐项清单：${escape(error.message)}。可在任务详情中查看已保留的跳过原因与样例。</td></tr>`;
      $("#exclusion-count").textContent = "清单暂不可用";
      $("#exclusion-prev").disabled = true; $("#exclusion-next").disabled = true;
    }
  }

  function batchMetadata(batch) {
    const keys = ["id", "job_id", "status", "items", "item_count", "characters", "elapsed_ms", "http_status", "boot_id", "created_at", "started_at", "finished_at", "error_code", "request_id", "attempt", "chunk_ids", "input_sha", "input_sha256", "input_sha256s", "input_hash", "input_hashes", "request_sha256"];
    return Object.fromEntries(keys.filter((key) => batch[key] !== undefined).map((key) => [key, key === "items" && Array.isArray(batch[key]) ? batch[key].length : batch[key]]));
  }

  function renderBatchList(batches) {
    if (!batches.length) return `<div class="plain-empty">尚无已持久化的编码批次记录。扫描进度、服务就绪状态均不能证明 GPU 编码请求已经完成。</div>`;
    return batches.map((batch) => {
      const ok = ["completed", "succeeded", "success", "ready", "done", "ok"].includes(batch.status), failed = ["failed", "error"].includes(batch.status);
      const items = Array.isArray(batch.items) ? batch.items.length : batch.items ?? batch.item_count;
      return `<details class="batch-record"><summary><span class="status-pill ${ok ? "completed" : failed ? "failed" : "running"}">${escape(ok ? "编码成功" : failed ? "编码失败" : statusNames[batch.status] || batch.status || "已记录")}</span><span class="batch-record-main"><strong>${count(items)} 个片段 · ${count(batch.characters)} 字符</strong><small>${escape(time(batch.created_at || batch.started_at))}${batch.http_status ? ` · HTTP ${escape(batch.http_status)}` : ""}${batch.error_code ? ` · ${escape(batch.error_code)}` : ""}</small></span><span class="batch-latency">${batch.elapsed_ms != null ? `${number(batch.elapsed_ms).toFixed(0)} ms` : "耗时待记录"}</span><span class="batch-expand">⌄</span></summary><div class="batch-record-details"><p>以下为真实请求元数据与输入摘要，不展示编码正文。输入摘要用于核对同一批内容。</p><pre>${escape(JSON.stringify(batchMetadata(batch), null, 2))}</pre></div></details>`;
    }).join("");
  }

  function renderBatchSummary(batches, job) {
    const details = object(job?.details), stats = object(details.batch_stats || details.batches);
    const successful = details.batches_succeeded ?? details.batches_success ?? details.batch_success ?? details.successful_batches ?? stats.success ?? stats.succeeded;
    const failed = details.batches_failed ?? details.batch_failures ?? details.failed_batches ?? stats.failed;
    const last = Object.keys(object(details.last_batch)).length ? object(details.last_batch) : batches[0];
    return `<div><span>累计成功批次</span><strong>${successful !== undefined ? count(successful) : "未提供累计数"}</strong></div><div><span>累计失败批次</span><strong>${failed !== undefined ? count(failed) : "未提供累计数"}</strong></div><div><span>最近批次耗时</span><strong>${last?.elapsed_ms != null ? `${number(last.elapsed_ms).toFixed(0)} ms` : "等待真实记录"}</strong></div><p>当前显示最近 ${count(batches.length)} 条持久化记录；累计数仅采用服务端明确返回的统计。</p>`;
  }

  async function loadBatches() {
    const job = state.jobs?.job || state.overview?.job;
    if (!job?.id) { $("#batch-summary").innerHTML = ""; $("#latest-batches").innerHTML = `<div class="plain-empty">尚未创建索引任务，没有编码批次。</div>`; return; }
    try {
      const data = await api(`/batches?${new URLSearchParams({ job_id: job.id, limit: "30" })}`);
      if (String((state.jobs?.job || state.overview?.job)?.id) !== String(job.id)) return;
      state.batches = Array.isArray(data.items) ? data.items : [];
      const openIds = new Set($$("#latest-batches .batch-record[open]").map((el) => el.dataset.batchId));
      $("#batch-summary").innerHTML = renderBatchSummary(state.batches, job);
      $("#latest-batches").innerHTML = renderBatchList(state.batches);
      $$("#latest-batches .batch-record").forEach((el, index) => { el.dataset.batchId = String(state.batches[index]?.id ?? index); el.open = openIds.has(el.dataset.batchId); });
    } catch (error) { $("#latest-batches").innerHTML = `<div class="plain-empty">批次观测暂不可用：${escape(error.message)}。任务计数仍可查看；此处不补造历史编码记录。</div>`; $("#batch-summary").innerHTML = renderBatchSummary([], job); }
  }

  async function openJob(id) {
    const serial = ++state.detailSerial;
    openDialog("任务、批次与异常", "任务记录");
    let data, notice = "";
    try { data = await api(`/jobs/${encodeURIComponent(id)}`); }
    catch (error) { const cached = state.jobRecords.get(String(id)); if (!cached) { if (serial === state.detailSerial) $("#detail-body").innerHTML = `<div class="error-detail">${escape(error.message)}</div>`; return; } data = {job: cached}; notice = `详细接口暂不可用，以下展示最近读到的任务记录：${error.message}`; }
    if (serial !== state.detailSerial || !detail.open) return;
    const job = data.job || state.jobRecords.get(String(id));
    if (!job) { $("#detail-body").innerHTML = `<div class="plain-empty">任务记录不存在</div>`; return; }
    const batches = Array.isArray(data.batches) ? data.batches : [], errors = Array.isArray(data.errors) ? data.errors : [];
    $("#detail-body").innerHTML = `${notice ? `<div class="notice warning">${escape(notice)}</div>` : ""}<div class="current-job job-inspector">${renderCurrentJob(job, true)}</div><section class="detail-section"><div class="detail-section-heading"><h3>编码批次</h3><span>只读观测，不提交编码请求</span></div><div class="batch-summary">${renderBatchSummary(batches, job)}</div><div class="batch-list">${renderBatchList(batches)}</div></section><section class="detail-section"><div class="detail-section-heading"><h3>异常记录</h3><span>${count(errors.length)} 条已返回记录</span></div>${errors.length ? errors.map((error) => `<div class="job-error-record"><strong>${escape(error.path || error.error_code || error.code || "处理异常")}</strong><p>${escape(error.message || error.error || error.reason || "服务未提供异常说明")}</p><small>${escape(time(error.created_at))}</small></div>`).join("") : `<p class="detail-explanation">详细接口未返回逐项异常。请同时查看任务消息、异常计数与完整观测字段。</p>`}</section>`;
  }

  function evaluationJSON(value, depth = 0) {
    if (depth > 12) return "[嵌套层级超过显示上限]";
    if (Array.isArray(value)) return value.map((item) => evaluationJSON(item, depth + 1));
    if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).filter(([key, item]) => !["content", "embedding_text", "source", "source_text", "snippet", "values", "body", "full_text"].includes(key.toLowerCase()) && !(key.toLowerCase() === "vector" && (Array.isArray(item) || typeof item === "string"))).map(([key, item]) => [key, evaluationJSON(item, depth + 1)]));
    return value;
  }

  function auditBlock(title, value, open = false) { return `<details class="raw-json evaluation-json"${open ? " open" : ""}><summary>${escape(title)}</summary><pre>${escape(JSON.stringify(evaluationJSON(value ?? {}), null, 2))}</pre></details>`; }

  function metricValue(value, percentage = false) {
    if (value === null || value === undefined || typeof value !== "number" || !Number.isFinite(value)) return "—";
    return percentage ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  }

  function readableEvaluation(data) {
    const scope = object(data.scope), snapshot = object(scope.snapshot || data.snapshot), summary = object(data.summary || data.metrics);
    const description = typeof scope.description === "string" ? scope.description : typeof data.scope === "string" ? data.scope : "记录未提供评测范围说明，请展开原始数据核对。";
    const snapshotCounts = ["files", "chunks", "vectors"].every((key) => typeof snapshot[key] === "number" && Number.isFinite(snapshot[key]));
    const scopeCard = `<section class="evaluation-scope-card"><h3>这次评测覆盖什么</h3><p>${escape(description)}</p><div class="scope-chips"><span>${scope.benchmark_subset === true ? "实验子集" : "范围以本轮记录为准"}</span><span>${scope.whole_ai_code_quality === true ? "记录声明评测全库，仍需核对覆盖" : "不代表全 AiCode 质量"}</span>${snapshotCounts ? `<span>${count(snapshot.files)} 文件 · ${count(snapshot.vectors)} / ${count(snapshot.chunks)} 向量</span>` : ""}</div></section>`;
    const gate = object(data.gate), reasons = Array.isArray(gate.reasons) ? gate.reasons : typeof gate.reasons === "string" ? [gate.reasons] : [];
    const rejected = data.status === "blocked" || gate.accepted === false;
    const reasonNames = { no_answer_failures: "未能正确拒绝无答案问题", regression_mrr: "排序质量相对基准下降", regression_recall: "召回率相对基准下降", regression_recall_at_5: "前五条召回相对基准下降", insufficient_coverage: "评测覆盖不足", incomplete_vectors: "向量覆盖不完整" };
    const gateCard = data.gate !== undefined || rejected ? `<section class="evaluation-gate-card${rejected ? " rejected" : ""}"><h3>${rejected ? "本轮未通过门槛" : gate.accepted === true ? "本轮通过已设置的门槛" : "门槛结论尚未提供"}</h3>${reasons.length ? `<ul>${reasons.map((reason) => { const code = typeof reason === "string" ? reason : JSON.stringify(reason); return `<li><span>${escape(reasonNames[code.split(":").pop()] || code)}</span>${reasonNames[code.split(":").pop()] ? `<code>${escape(code)}</code>` : ""}</li>`; }).join("")}</ul>` : `<p>${rejected ? "服务端未提供具体原因，可展开门槛原始记录核对。" : "结论仅适用于本轮数据和门槛，不等于全库检索质量已获证明。"}</p>`}</section>` : "";
    const modeEntries = ["bm25", "vector", "hybrid"].map((mode) => [mode, object(summary[mode])]);
    const hasMetrics = modeEntries.some(([, value]) => Object.keys(value).length);
    let metricsCard = `<section class="evaluation-metrics-card"><h3>各模式的实际记录</h3><div class="plain-empty"><strong>暂无可比较的指标</strong>空记录表示尚未提供，不能当成零分或已通过。</div></section>`;
    if (hasMetrics) {
      const rows = modeEntries.flatMap(([mode, values]) => {
        const splits = ["development", "holdout"].filter((split) => values[split] && typeof values[split] === "object");
        return (splits.length ? splits : [null]).map((split) => {
          const metrics = split ? object(values[split]) : values;
          const cases = typeof metrics.cases === "number" ? metrics.cases : null, scorable = typeof metrics.scorable === "number" ? metrics.scorable : null;
          const missing = !Object.keys(metrics).length, unrun = cases === 0, unscorable = scorable === 0;
          const issues = [[metrics.degraded_cases, "降级"], [metrics.incomplete_semantic_cases, "覆盖不足"], [metrics.execution_error_cases, "执行失败"], [metrics.skipped, "跳过"]].filter(([value]) => typeof value === "number" && value > 0);
          const execution = missing ? "未提供记录" : unrun ? "未执行" : unscorable ? "无可评分案例" : `${cases === null ? "执行数量未提供" : `${count(cases)} 题`}${scorable !== null && !unrun ? ` · 可评分 ${count(scorable)}` : ""}`;
          const value = (key, percentage = false) => missing || unrun || unscorable ? "—" : metricValue(metrics[key], percentage);
          return `<tr><th scope="row">${escape(modeNames[mode])}</th><td>${split === "development" ? "开发集" : split === "holdout" ? "留出集" : "当前记录"}</td><td class="execution-cell">${escape(execution)}${issues.length ? `<small>${issues.map(([value, label]) => `${count(value)} ${label}`).join(" · ")}</small>` : ""}</td><td class="numeric">${value("mrr")}</td><td class="numeric">${value("recall@5", true)}</td><td class="numeric">${value("recall@10", true)}</td><td class="numeric">${value("no_answer_accuracy", true)}</td></tr>`;
        });
      }).join("");
      metricsCard = `<section class="evaluation-metrics-card"><h3>各模式的实际记录</h3><div class="table-scroll"><table class="evaluation-metrics-table"><thead><tr><th>检索方式</th><th>数据集</th><th>执行情况</th><th class="numeric">MRR</th><th class="numeric">Recall@5</th><th class="numeric">Recall@10</th><th class="numeric">无答案准确率</th></tr></thead><tbody>${rows}</tbody></table></div><p>“—”表示未提供或未执行；数值 0 只在记录明确给出时展示。覆盖不足、跳过与执行失败请结合逐题证据判断。</p></section>`;
    }
    return scopeCard + gateCard + metricsCard;
  }

  async function loadEvaluations() {
    try {
      const data = await api("/evaluations");
      const items = Array.isArray(data.items) ? data.items : [];
      const selected = state.selectedEvaluation;
      $("#evaluation-count").textContent = `${count(items.length)} 条运行 / 试验`;
      $("#evaluation-run").innerHTML = items.length ? items.map((item) => `<option value="${escape(item.id || item.run_id)}">${escape(item.id || item.run_id)} · ${escape(statusNames[item.status] || item.status || "状态未记录")}</option>`).join("") : `<option value="">尚无运行记录</option>`;
      const latest = typeof data.latest === "string" ? data.latest : data.latest?.id || data.latest?.run_id;
      const id = items.some((item) => String(item.id || item.run_id) === selected) ? selected : items.some((item) => String(item.id || item.run_id) === String(latest)) ? latest : items[0]?.id || items[0]?.run_id;
      if (id) { selectValue("#evaluation-run", id); await loadEvaluation(String(id)); }
      else { state.selectedEvaluation = ""; $("#evaluation-detail").innerHTML = `<div class="plain-empty"><strong>暂无已保存的评测运行</strong>完成数据准备或评测后，这里会展示真实运行状态、参数与逐题证据。</div>`; }
    } catch (error) { $("#evaluation-count").textContent = "读取失败"; $("#evaluation-detail").innerHTML = `<div class="notice error">无法读取评测记录：${escape(error.message)}</div>`; }
  }

  async function loadEvaluation(id) {
    const serial = ++state.evaluationSerial;
    const openEvidence = state.selectedEvaluation === id ? new Set($$("#evaluation-detail .evaluation-json[open]").map((el) => $("summary", el)?.textContent)) : new Set();
    state.selectedEvaluation = id;
    $("#evaluation-detail").setAttribute("aria-busy", "true");
    try {
      const data = await api(`/evaluations/${encodeURIComponent(id)}`);
      if (serial !== state.evaluationSerial) return;
      const status = data.status || "未记录", cases = data.cases ?? [], events = Array.isArray(data.events) ? data.events.slice(-100) : data.events ?? [];
      const gate = data.gate !== undefined ? auditBlock("门槛原始记录", data.gate) : "";
      const statusClass = ["failed", "error"].includes(status) ? "failed" : ["blocked", "partial", "paused", "interrupted", "encoded", "built"].includes(status) ? "partial" : ["accepted", "completed", "succeeded", "done"].includes(status) ? "completed" : "";
      $("#evaluation-detail").innerHTML = `<div class="evaluation-run-heading"><div><span class="eyebrow">评测记录</span><h2>${escape(data.id || data.run_id || id)}</h2><p>${data.created_at ? `创建于 ${escape(time(data.created_at))} · 北京时间` : "创建时间未提供"}</p></div><span class="status-pill ${statusClass}">${escape(statusNames[status] || status)}</span></div>${readableEvaluation(data)}<div class="section-label"><h3>展开核对完整证据</h3><span class="muted">原始记录保留，不展示正文</span></div><div class="evaluation-panels">${gate}${auditBlock("结果与指标原始记录", data.summary ?? data.metrics ?? {})}${auditBlock("数据范围与覆盖状态", data.scope ?? {})}${auditBlock("查询参数与试验配置", data.parameters ?? {})}${auditBlock(`逐题证据 · ${Array.isArray(cases) ? count(cases.length) : "按记录"}`, cases)}${auditBlock("运行过程 · 最近 100 条事件", events)}</div>`;
      $$("#evaluation-detail .evaluation-json").forEach((el) => { el.open = openEvidence.has($("summary", el)?.textContent); });
    } catch (error) { if (serial === state.evaluationSerial) $("#evaluation-detail").innerHTML = `<div class="notice error">无法读取运行 ${escape(id)}：${escape(error.message)}</div>`; }
    finally { if (serial === state.evaluationSerial) $("#evaluation-detail").setAttribute("aria-busy", "false"); }
  }

  function updateActionButtons() {
    const running = state.submitting || active(state.jobs?.job || state.overview?.job);
    for (const selector of ["#sync-button", "#scan-button", "#embed-button"]) $(selector).disabled = running;
  }

  async function startJob(action) {
    if (state.submitting) return;
    state.submitting = true;
    updateActionButtons();
    try {
      const job = await api("/jobs", { method: "POST", body: JSON.stringify({ action }) });
      state.jobs = { ...(state.jobs || {}), job };
      if (state.overview) state.overview.job = job;
      toast(`${actionNames[action]}任务已提交，可在此查看实际进度。`);
      setView("jobs");
      renderJobBanner(job);
      $("#current-job").innerHTML = renderCurrentJob(job);
      schedulePoll(1200);
    } catch (error) { toast(`任务未启动：${error.message}`); }
    finally { state.submitting = false; updateActionButtons(); }
  }

  async function cancelJob(button) {
    button.disabled = true;
    try {
      const job = await api("/jobs/cancel", { method: "POST" });
      state.jobs = { ...(state.jobs || {}), job };
      if (state.overview) state.overview.job = job;
      $("#current-job").innerHTML = renderCurrentJob(job);
      renderJobBanner(job); updateActionButtons();
      toast("已发送停止请求；当前请求结束后，任务会保存已完成的进度。");
    } catch (error) { button.disabled = false; toast(error.message); }
  }

  async function loadTraces() {
    try {
      const data = await api("/traces?limit=12");
      const items = Array.isArray(data.items) ? data.items : [];
      state.traces.clear(); items.forEach((item, index) => state.traces.set(String(item.id || item.trace_id || index), item));
      $("#recent-traces").innerHTML = items.length ? items.map((item, index) => `<div class="recent-row"><button type="button" class="trace-replay" data-trace="${escape(item.id || item.trace_id || index)}" aria-label="恢复原参数并重新检索 ${escape(item.query)}">${icon("search")}<span>${escape(item.query)}</span><small>${escape(modeNames[item.mode] || item.mode)} · ${number(item.elapsed_ms).toFixed(0)} ms${item.parameters?.project ? ` · ${escape(item.parameters.project)}` : ""}</small>${icon("arrow")}</button></div>`).join("") : `<div class="plain-empty">搜索后，可在这里查看真实检索记录，并恢复原参数重新运行。</div>`;
    } catch (error) { $("#recent-traces").innerHTML = `<div class="plain-empty">${escape(error.message)}</div>`; }
  }

  function openDialog(title, eyebrow = "原文与依据") {
    $("#detail-title").textContent = title;
    $("#detail-eyebrow").textContent = eyebrow;
    $("#detail-body").innerHTML = `<div class="results-loading"><span class="spinner" aria-hidden="true"></span>正在读取详情…</div>`;
    if (!detail.open) detail.showModal();
    $("#detail-body").scrollTop = 0;
    document.body.style.overflow = "hidden";
  }

  async function openChunk(id) {
    const serial = ++state.detailSerial;
    const result = state.results.get(String(id));
    openDialog("原文与召回详情");
    try {
      const data = await api(`/chunks/${encodeURIComponent(id)}`);
      if (serial !== state.detailSerial || !detail.open) return;
      const chunk = data.chunk || {}, vector = data.vector;
      state.detailData = { ...data, result };
      const content = String(chunk.content || "");
      const context = Array.isArray(data.context) && data.context.length ? data.context : content.split("\n").map((text, index) => ({ number: number(chunk.start_line) + index, text }));
      $("#detail-body").innerHTML = pathBox(chunk.path || data.file?.path, content) + `<div class="detail-chips"><span>${escape(chunk.project || "根目录")}</span><span>L${count(chunk.start_line)}–${count(chunk.end_line)}</span><span>片段 ${escape(chunk.id || id)}</span><span>${vector ? "向量已生成" : "尚无向量"}</span></div>` + (result ? routeDetails(result) : `<p class="detail-explanation">当前从资料库查看原文。检索后打开结果，可以对照该查询的两路排名与得分。</p>`) + `<section class="detail-section"><div class="detail-section-heading"><h3>原文上下文</h3><button class="text-button" type="button" data-copy-content>${icon("copy")}复制片段</button></div><div class="source-view" tabindex="0" role="region" aria-label="原文上下文，可横向滚动">${context.map((line) => `<div class="source-line${number(line.number) >= number(chunk.start_line) && number(line.number) <= number(chunk.end_line) ? " in-chunk" : ""}"><span class="source-line-number">${count(line.number)}</span><span class="source-line-text">${highlight(line.text, result?.matched_terms) || " "}</span></div>`).join("")}</div><div class="context-legend"><i></i>蓝色背景表示当前召回片段，行号对应索引时的原文。</div></section>` + vectorDetails(vector) + `<details class="raw-json"><summary>完整片段与元数据 JSON</summary><pre>${escape(JSON.stringify({ chunk, file: data.file, vector: vector ? { ...vector, values: undefined } : null, retrieval: result || undefined }, null, 2))}</pre></details>` + (vector?.values ? `<details class="raw-json"><summary>查看完整向量 · ${count(vector.values.length)} 维</summary><pre>${escape(JSON.stringify(vector.values))}</pre></details>` : "");
    } catch (error) { if (serial === state.detailSerial) $("#detail-body").innerHTML = `<div class="error-detail">${escape(error.message)}</div>`; }
  }

  function pathBox(path, content) {
    state.copyPath = path || "";
    state.copyContent = content || "";
    return `<div class="detail-path">${icon("file")}<span>${escape(path || "路径未记录")}</span><button class="icon-button" type="button" data-copy-path aria-label="复制文件路径">${icon("copy")}</button></div>`;
  }

  function routeDetails(result) {
    const explanation = state.search?.explanation || {};
    const fusion = (state.search?.effective_mode || state.search?.mode) === "hybrid";
    const vectorContribution = number(result.vector_contribution), bm25Contribution = number(result.bm25_contribution);
    const sum = vectorContribution + bm25Contribution;
    return `<section class="detail-section"><div class="detail-section-heading"><h3>召回依据</h3><span>${escape(modeNames[state.search?.effective_mode || state.search?.mode] || "检索")}</span></div><table class="route-table"><thead><tr><th>召回通路</th><th>通路排名</th><th>原始得分</th><th>${fusion ? "RRF 贡献" : "贡献"}</th></tr></thead><tbody><tr><td>语义向量</td><td>${ranks(result.vector_rank)}</td><td>${score(result.vector_score, 5)}</td><td>${score(result.vector_contribution, 7)}</td></tr><tr><td>${result.lexical_score != null ? "词项路（BM25＋重排）" : "BM25 关键词"}</td><td>${ranks(result.bm25_rank)}</td><td><span class="score-stack">${score(result.raw_bm25_score ?? result.bm25_raw_score ?? result.bm25_score, 5)}${result.lexical_score != null ? `<small>重排 ${score(result.lexical_score, 5)}</small>` : ""}</span></td><td>${score(result.bm25_contribution, 7)}</td></tr></tbody><tfoot><tr><td colspan="3">${fusion ? "融合得分" : "最终得分"}</td><td>${score(result.score, 7)}</td></tr></tfoot></table>${fusion && sum > 0 ? `<div class="contribution-chart" aria-label="语义贡献 ${score(vectorContribution / sum * 100, 1)}%，关键词贡献 ${score(bm25Contribution / sum * 100, 1)}%"><span class="vector" style="width:${vectorContribution / sum * 100}%"></span><span class="bm25" style="width:${bm25Contribution / sum * 100}%"></span></div>` : ""}<p class="detail-explanation">向量相似度和 BM25 的量纲不同，不能直接比较大小。词项路先使用 BM25 召回，再按公开特征重排；RRF 融合重排后的通路排名。${fusion ? "混合检索使用排名融合，分数表示当前查询中的排序依据。" : "分数用于当前查询排序，不代表答案正确的概率。"}</p>${fusion ? `<div class="formula-box">${escape(explanation.formula || "score = w_vector / (k + rank_vector) + w_bm25 / (k + rank_bm25)")}<br>k = ${escape(explanation.rrf_k ?? "—")} · w_vector = ${score(explanation.vector_weight, 2)} · w_bm25 = ${score(explanation.bm25_weight, 2)} · 每路候选 ${count(explanation.candidates)}</div>` : ""}</section>`;
  }

  function vectorDetails(vector) {
    if (!vector) return `<section class="detail-section"><div class="detail-section-heading"><h3>向量白盒</h3></div><div class="plain-empty" style="border:1px dashed #e6e6e6;border-radius:6px;padding:25px 16px">此片段尚无向量，完成向量化后可参与语义召回。<br>可在资料库点击「补齐向量」，查看任务的真实进度。</div></section>`;
    const values = Array.isArray(vector.values) ? vector.values.map(Number).filter(Number.isFinite) : [];
    const sampled = values.length > 180 ? Array.from({ length: 180 }, (_, i) => values[Math.round(i * (values.length - 1) / 179)]) : values;
    const maxAbs = Math.max(...sampled.map(Math.abs), .000001);
    const path = sampled.map((v, i) => `${i ? "L" : "M"}${(i / Math.max(1, sampled.length - 1) * 600).toFixed(2)},${(40 - v / maxAbs * 31).toFixed(2)}`).join(" ");
    return `<section class="detail-section"><div class="detail-section-heading"><h3>向量白盒</h3><span>已持久化到本地 SQLite</span></div><div class="vector-summary"><div>维度<strong>${count(vector.dimensions || values.length)}</strong></div><div>L2 范数<strong>${score(vector.norm, 6)}</strong></div><div>模型版本<strong title="${escape(vector.revision || "")}">${escape(vector.revision ? String(vector.revision).slice(0, 12) : "未标记")}</strong></div></div>${sampled.length ? `<svg class="vector-sparkline" viewBox="0 0 600 80" preserveAspectRatio="none" role="img" aria-label="真实向量分量曲线"><line x1="0" y1="40" x2="600" y2="40"/><path d="${path}"/></svg><p class="detail-explanation">${values.length > 180 ? `等距采样 ${count(sampled.length)} 个真实分量，` : `显示全部 ${count(values.length)} 个分量，`}纵轴按最大绝对值缩放；下方可展开完整数值。<br>模型：${escape(vector.model || "未记录")}</p>` : `<p class="detail-explanation">服务未返回向量数值。</p>`}</section>`;
  }

  async function openFile(id) {
    const serial = ++state.detailSerial;
    openDialog("文件与索引片段");
    try {
      const data = await api(`/files/${encodeURIComponent(id)}`);
      if (serial !== state.detailSerial || !detail.open) return;
      const file = data.file || {}, chunks = Array.isArray(data.chunks) ? data.chunks : [];
      const content = String(data.content || "");
      state.detailData = data;
      const vectorCount = file.vector_count ?? chunks.filter((chunk) => chunk.vector_status === "ready").length;
      $("#detail-body").innerHTML = pathBox(file.path, content) + `<div class="detail-chips"><span>${escape(file.project || "根目录")}</span><span>${escape(file.extension || "无扩展名")}</span><span>${formatBytes(file.size)}</span><span>${count(chunks.length)} 个片段</span><span>${count(vectorCount)} 个向量</span></div>${file.error ? `<div class="error-detail">${escape(file.error)}</div>` : ""}<section class="detail-section"><div class="detail-section-heading"><h3>索引片段</h3><span>点击检查内容与向量</span></div><div class="chunk-list">${chunks.length ? chunks.map((chunk, index) => `<button type="button" class="chunk-button" data-chunk="${escape(chunk.id)}"><span><strong>片段 ${count(index + 1)} · L${count(chunk.start_line)}–${count(chunk.end_line)} <span class="status-pill">${chunk.vector_status === "ready" ? "向量就绪" : chunk.vector_status === "pending" ? "待向量化" : escape(chunk.vector_status || "—")}</span></strong><small>${escape(String(chunk.content || "").slice(0, 170).replace(/\s+/g, " "))}</small></span>${icon("chevron")}</button>`).join("") : `<div class="plain-empty">该文件没有可检索片段。</div>`}</div></section><section class="detail-section"><div class="detail-section-heading"><h3>文件原文</h3><button class="text-button" type="button" data-copy-content>${icon("copy")}复制原文</button></div><pre class="detail-content-pre">${escape(content || "服务未返回文件原文。")}</pre></section><details class="raw-json"><summary>文件元数据 JSON</summary><pre>${escape(JSON.stringify(file, null, 2))}</pre></details>`;
    } catch (error) { if (serial === state.detailSerial) $("#detail-body").innerHTML = `<div class="error-detail">${escape(error.message)}</div>`; }
  }

  function showTrace() {
    if (!state.search) return;
    ++state.detailSerial;
    openDialog("检索链路", "本次检索依据");
    const data = state.search;
    const explanation = data.explanation || {};
    $("#detail-body").innerHTML = `<p class="trace-intro">查看这次查询实际使用的模式、候选数量与融合参数。每条结果的原文详情包含对应通路的排名、得分与贡献。</p><div class="trace-query">${escape(data.query || data.request.query)}</div><div class="trace-summary"><div>实际模式<strong>${escape(modeNames[data.effective_mode || data.mode] || data.effective_mode || data.mode)}</strong></div><div>总候选<strong>${count(data.total_candidates)}</strong></div><div>查询耗时<strong>${number(data.elapsed_ms).toFixed(0)} ms</strong></div></div>${data.warning ? `<div class="notice warning">${escape(data.warning)}</div>` : ""}<div class="detail-section-heading"><h3>检索过程</h3></div><div class="formula-box">1. 过滤项目与文件类型<br>2. ${(data.effective_mode || data.mode) === "bm25" ? "BM25 词项检索" : (data.effective_mode || data.mode) === "vector" ? "查询向量化 → 语义相似度排序" : "查询向量化 → 语义召回 + BM25 词项召回 → RRF 融合"}<br>3. 返回前 ${count(data.request.limit)} 条片段与来源信息</div><section class="detail-section trace-timings"><div class="detail-section-heading"><h3>通路耗时</h3></div><div class="vector-summary"><div>BM25 词项召回<strong>${explanation.timings?.bm25_ms !== undefined ? `${number(explanation.timings.bm25_ms).toFixed(1)} ms` : "未执行"}</strong></div><div>语义向量链路<strong>${explanation.timings?.vector_ms !== undefined ? `${number(explanation.timings.vector_ms).toFixed(1)} ms` : "未执行"}</strong></div><div>服务端检索<strong>${number(data.elapsed_ms).toFixed(1)} ms</strong></div></div><p class="detail-explanation">语义链路耗时包含 AutoDL 查询编码与本地精确余弦扫描。当前实现逐批扫描 SQLite 中的向量，按项目过滤可缩小扫描范围。</p></section><section class="detail-section"><div class="detail-section-heading"><h3>本次请求与服务响应</h3><span>Trace ${escape(data.trace_id || "未记录")}</span></div><details class="raw-json" open><summary>请求参数</summary><pre>${escape(JSON.stringify(data.request, null, 2))}</pre></details><details class="raw-json" open><summary>评分与范围</summary><pre>${escape(JSON.stringify(explanation, null, 2))}</pre></details><details class="raw-json"><summary>完整响应 JSON</summary><pre>${escape(JSON.stringify({ ...data, request: undefined }, null, 2))}</pre></details></section>`;
  }

  async function copy(text, success) {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(String(text || ""));
      else {
        const field = document.createElement("textarea"); field.value = String(text || ""); field.style.cssText = "position:fixed;opacity:0;pointer-events:none";
        (detail.open ? detail : document.body).appendChild(field); field.select();
        const copied = document.execCommand("copy"); field.remove();
        if (!copied) throw new Error("复制不可用");
      }
      toast(success);
    } catch { toast("浏览器未允许复制，请在原文中手动选择内容。"); }
  }

  function closeSidebar() { $("#sidebar").classList.remove("open"); $("#sidebar-backdrop").hidden = true; $("#mobile-menu").setAttribute("aria-expanded", "false"); }

  async function poll() {
    if (state.polling) { schedulePoll(4000); return; }
    if (document.hidden) { schedulePoll(15000); return; }
    state.polling = true;
    const wasActive = active(state.jobs?.job || state.overview?.job);
    const results = await Promise.allSettled([loadOverview(), loadJobs(), loadHealth()]);
    if (results[0].status === "fulfilled" && results[1].status === "fulfilled") globalError("");
    else if (results[0].status === "rejected" && results[1].status === "rejected") globalError(`暂时无法连接本地服务：${results[0].reason.message}。正在定期重试。`);
    const isActive = active(state.jobs?.job || state.overview?.job);
    if (state.view === "jobs") await loadBatches();
    if (state.view === "evaluations") await loadEvaluations();
    if (wasActive && !isActive) {
      if (state.view === "library") loadLibrary();
      toast("索引任务状态已更新，可在任务页查看处理结果。");
    }
    state.polling = false;
    schedulePoll(isActive ? 4000 : 20000);
  }

  function schedulePoll(delay) { clearTimeout(state.pollTimer); state.pollTimer = setTimeout(poll, delay); }

  $("#search-form").addEventListener("submit", (event) => { event.preventDefault(); runSearch(); });
  $("#search-form").addEventListener("input", syncSearchDraft);
  $("#search-form").addEventListener("change", syncSearchDraft);
  $$(".mode-button").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault(); const buttons = $$(".mode-button"), index = buttons.indexOf(button);
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
      setMode(buttons[next].dataset.mode); buttons[next].focus();
    });
  });
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  $("#project-filter").addEventListener("change", (event) => setProject(event.target.value));
  $("#library-project").addEventListener("change", () => { state.libraryOffset = 0; loadLibrary(); });
  $("#vector-weight").addEventListener("input", (event) => { $("#weight-value").textContent = `${event.target.value}%`; });
  $("#sync-button").addEventListener("click", () => startJob("sync"));
  $("#scan-button").addEventListener("click", () => startJob("scan"));
  $("#embed-button").addEventListener("click", () => startJob("embed"));
  $("#library-refresh").addEventListener("click", () => Promise.allSettled([loadOverview(), loadLibrary()]).then((results) => { const failure = results.find((r) => r.status === "rejected"); if (failure) toast(failure.reason.message); }));
  $("#health-button").addEventListener("click", () => loadHealth().then(() => toast(state.health.message || "服务状态已更新")).catch((error) => toast(error.message)));
  $("#jobs-refresh").addEventListener("click", () => Promise.allSettled([loadJobs(), loadOverview()]).then((results) => { const failure = results.find((r) => r.status === "rejected"); toast(failure ? failure.reason.message : "任务进度已更新"); }));
  $("#traces-refresh").addEventListener("click", loadTraces);
  $("#batches-refresh").addEventListener("click", loadBatches);
  $("#evaluations-refresh").addEventListener("click", loadEvaluations);
  $("#evaluation-run").addEventListener("change", (event) => { if (event.target.value) loadEvaluation(event.target.value); });
  $("#exclusion-refresh").addEventListener("click", () => Promise.allSettled([loadOverview(), loadExclusions()]));
  $("#exclusion-reason").addEventListener("change", () => { state.exclusionOffset = 0; loadExclusions(); });
  $("#exclusion-prev").addEventListener("click", () => { state.exclusionOffset = Math.max(0, state.exclusionOffset - 50); loadExclusions(); });
  $("#exclusion-next").addEventListener("click", () => { if (state.exclusionOffset + 50 < state.exclusionTotal) { state.exclusionOffset += 50; loadExclusions(); } });
  let exclusionTimer;
  for (const selector of ["#exclusion-query", "#exclusion-project"]) $(selector).addEventListener("input", () => { clearTimeout(exclusionTimer); exclusionTimer = setTimeout(() => { state.exclusionOffset = 0; loadExclusions(); }, 280); });
  $("#show-trace").addEventListener("click", showTrace);
  $("#library-prev").addEventListener("click", () => { state.libraryOffset = Math.max(0, state.libraryOffset - 50); loadLibrary(); });
  $("#library-next").addEventListener("click", () => { if (state.libraryOffset + 50 < state.libraryTotal) { state.libraryOffset += 50; loadLibrary(); } });
  let libraryTimer;
  $("#library-query").addEventListener("input", () => { clearTimeout(libraryTimer); libraryTimer = setTimeout(() => { state.libraryOffset = 0; loadLibrary(); }, 280); });
  $("#detail-close").addEventListener("click", () => detail.close());
  detail.addEventListener("close", () => { state.detailSerial++; document.body.style.overflow = ""; });
  detail.addEventListener("click", (event) => { if (event.target === detail && event.clientX < detail.getBoundingClientRect().left) detail.close(); });
  $("#mobile-menu").addEventListener("click", () => { const opened = $("#sidebar").classList.toggle("open"); $("#sidebar-backdrop").hidden = !opened; $("#mobile-menu").setAttribute("aria-expanded", String(opened)); });
  $("#sidebar-backdrop").addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { closeSidebar(); const settings = $(".advanced-controls"); if (!detail.open && settings.open) { settings.open = false; if (settings.contains(document.activeElement)) $("summary", settings).focus(); } }
    const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
    if ((event.key === "/" && !editing && !event.ctrlKey && !event.metaKey) || (event.key.toLowerCase() === "k" && (event.ctrlKey || event.metaKey))) {
      if (detail.open) return;
      event.preventDefault(); setView("search"); $("#query").focus();
    }
  });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("button"); if (!button) return;
    if (button.hasAttribute("data-project")) setProject(button.dataset.project, { navigate: true });
    if (button.hasAttribute("data-query")) { if (button.dataset.replayMode) setMode(button.dataset.replayMode); setView("search"); $("#query").value = button.dataset.query; runSearch(); }
    if (button.hasAttribute("data-trace")) { const trace = state.traces.get(String(button.dataset.trace)); if (trace) { setView("search"); restoreSearch({ ...object(trace.parameters || trace.request), query: trace.query, mode: trace.mode }); runSearch(); } }
    if (button.hasAttribute("data-chunk")) openChunk(button.dataset.chunk);
    if (button.hasAttribute("data-file")) openFile(button.dataset.file);
    if (button.hasAttribute("data-open-jobs")) setView("jobs");
    if (button.hasAttribute("data-open-library")) setView("library");
    if (button.hasAttribute("data-open-exclusions")) setView("exclusions");
    if (button.hasAttribute("data-job")) openJob(button.dataset.job);
    if (button.hasAttribute("data-retry-search")) runSearch();
    if (button.hasAttribute("data-start-job")) startJob(button.dataset.startJob);
    if (button.hasAttribute("data-cancel-job")) cancelJob(button);
    if (button.hasAttribute("data-copy-path")) copy(state.copyPath, "文件路径已复制");
    if (button.hasAttribute("data-copy-content")) copy(state.copyContent, "原文内容已复制");
  });
  document.addEventListener("click", (event) => { const settings = $(".advanced-controls"); if (settings.open && !settings.contains(event.target)) settings.open = false; });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) schedulePoll(100); });

  async function init() {
    const params = new URLSearchParams(location.search);
    const results = await Promise.allSettled([loadOverview(), loadJobs(), loadHealth()]);
    if (results[0].status === "rejected") {
      globalError(`无法读取本地资料库：${results[0].reason.message}。请确认服务正在运行，页面将自动重试。`);
      $("#project-list").innerHTML = `<div class="sidebar-loading">目录暂不可读</div>`;
      $("#storage-path").textContent = "等待本地服务连接";
      $("#library-root").textContent = "等待本地服务连接";
    }
    if (results[1].status === "rejected") $("#current-job").innerHTML = `<div class="plain-empty">${escape(results[1].reason.message)}</div>`;
    restoreSearch({ query: params.get("q") || params.get("query") || "", mode: params.get("mode"), project: params.get("project"), extension: params.get("extension"), vector_weight: params.has("vector_weight") ? params.get("vector_weight") : undefined, limit: params.get("limit"), candidates: params.get("candidates") });
    if (params.get("q") || params.get("query")) runSearch();
    schedulePoll(active(state.jobs?.job || state.overview?.job) ? 4000 : 20000);
  }
  init();
})();
