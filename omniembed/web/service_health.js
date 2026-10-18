// 页面仅读取历史。只有明确操作监测开关时，才修改 Web 后台的采样设置。
(() => {
  const panel = $("#tab-health");
  const refreshButton = $("#health-refresh");
  const toggleButton = $("#health-toggle");
  const profileButtons = [...panel.querySelectorAll("[data-health-profile]")];
  const statusLabels = {
    HEALTHY: "健康", RECOVERING: "恢复确认", BUSY: "忙碌", STARTING: "启动中",
    DEGRADED: "异常待确认", DOWN: "不可用", AUTH_ERROR: "鉴权失败",
    CONFIG_ERROR: "监测配置异常", STALE: "监测停滞", UNKNOWN: "未知",
    MONITOR_ERROR: "监测器异常", EXPIRED: "服务已到期", STOPPED: "服务已停止", NOT_CONFIGURED: "尚未配置",
    RELEASE_UNCONFIRMED: "资源释放未确认", DISABLED: "已停用",
  };
  const reasonLabels = {
    checks_passed: "检查通过", model_not_ready: "模型尚未就绪", configuration_error: "监测配置异常",
    ready_inference_stalled: "推理超时，等待进程恢复", model_degraded: "模型推理异常",
    model_degraded_pending_probe: "等待深度复查", inference_probe_pending_retry: "等待下轮深度复查",
    inference_probe_unauthorized: "深度检查鉴权失败", live_unauthorized: "存活检查鉴权失败",
    ready_unauthorized: "就绪检查鉴权失败", inference_unauthorized: "深度检查鉴权失败",
    live_busy: "存活检查忙碌", ready_busy: "就绪检查忙碌", inference_busy: "推理忙碌",
    live_check_failed: "存活检查失败", ready_check_failed: "就绪检查失败", inference_check_failed: "深度检查失败",
    inference_invalid_self_test: "向量自检未通过",
  };
  const unavailableLabels = {
    not_configured: "尚未配置服务地址或访问密钥。",
    expired: "原有 Colab 租约已到期，需先恢复已有部署并更新连接配置。",
    stopped: "原有 Colab 实例已停止，需先恢复部署并更新连接配置。",
    release_unconfirmed: "原有 Colab 资源的释放状态尚未确认，请先确认实例状态。",
    config_error: "服务监测配置需要修复，请检查地址、访问密钥及配置文件。",
  };
  let timer = null;
  let activeRequest = null;
  let requestNumber = 0;
  let mutationBusy = false;
  let selectedProfile = null;
  let lastSummary = null;
  let lastReadAt = null;
  let archiveState = null;
  let readSavedSelection = false;

  const visible = () => !panel.hidden && !document.hidden;
  const numeric = value => value == null || value === "" || !Number.isFinite(Number(value)) ? null : Number(value);
  const count = value => Math.max(0, Math.trunc(numeric(value) ?? 0));
  const text = (selector, value) => { $(selector).textContent = value; };
  const localTime = value => {
    if (!value) return "—";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? "—" : parsed.toLocaleString("zh-CN", {
      year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
      second: "2-digit", hour12: false, timeZone: "Asia/Shanghai",
    });
  };
  const milliseconds = value => numeric(value) == null ? "—" : `${Number(value).toFixed(1)} ms`;
  const ageLabel = value => {
    const seconds = Math.max(0, Math.floor(value));
    if (seconds < 60) return `${seconds} 秒前`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒前`;
    return `${Math.floor(seconds / 3600)} 小时 ${Math.floor(seconds % 3600 / 60)} 分前`;
  };
  const category = status => status === "HEALTHY" ? "good" : ["DOWN", "AUTH_ERROR", "CONFIG_ERROR", "MONITOR_ERROR", "RELEASE_UNCONFIRMED"].includes(status) ? "bad" : "warning";
  const tableStatus = status => {
    const normalized = String(status || "UNKNOWN").toUpperCase();
    const className = normalized === "HEALTHY" ? "ready" : ["DOWN", "AUTH_ERROR", "CONFIG_ERROR"].includes(normalized) ? "failed" : "embedding";
    return `<span class="st ${className}"><i aria-hidden="true"></i>${esc(statusLabels[normalized] || normalized)}</span>`;
  };
  const checkCell = check => {
    if (!check || typeof check !== "object") return '<td class="mute">本轮未执行</td>';
    const code = numeric(check.http_status);
    const label = code != null && code > 0 ? `HTTP ${code}` : "未收到响应";
    return `<td class="health-http">${esc(label)}<small>${esc(milliseconds(check.latency_ms))}</small></td>`;
  };
  const validProfile = id => id === "local" || id === "colab";
  const currentProfile = () => lastSummary?.profile;

  function syncControls() {
    refreshButton.disabled = !!activeRequest || mutationBusy;
    const profile = currentProfile();
    toggleButton.disabled = !profile || mutationBusy || (!profile.enabled && profile.can_enable !== true);
    profileButtons.forEach(button => { button.disabled = mutationBusy; });
  }

  function renderProfile(summary) {
    const profile = summary.profile;
    const disabled = profile.enabled !== true;
    const label = profile.label || (profile.id === "colab" ? "Google Colab" : "本地电脑");
    panel.classList.toggle("health-disabled", disabled);
    $("#health-profile-content").setAttribute("aria-labelledby", `health-profile-${profile.id}`);
    text("#health-title", `${label}健康监测`);
    text("#health-target-url", profile.url || "尚未配置监测地址");
    text("#health-toggle-state", profile.stopping ? "正在停止" : disabled ? "已停用" : "已启用");
    toggleButton.setAttribute("aria-checked", String(!disabled));
    toggleButton.setAttribute("aria-label", `启用${profile.id === "colab" ? "Colab" : "本地电脑"}监测`);
    const profiles = Array.isArray(summary.profiles) ? summary.profiles : [profile];
    profileButtons.forEach(button => {
      const id = button.dataset.healthProfile;
      const selected = id === profile.id;
      const item = profiles.find(value => value.id === id);
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      text(`#health-profile-${id}-state`, item ? item.enabled ? "已启用" : "已停用" : "未配置");
    });
    const openLink = $("#health-open-service");
    let safeUrl = null;
    if (profile.id === "local" && profile.url) {
      try {
        const url = new URL(profile.url);
        if (["http:", "https:"].includes(url.protocol) && !url.username && !url.password) safeUrl = url.href;
      } catch { /* 未配置合法地址时不显示链接。 */ }
    }
    openLink.hidden = !safeUrl;
    if (safeUrl) openLink.href = safeUrl;
    else openLink.removeAttribute("href");

    const unavailable = unavailableLabels[profile.unavailable_reason] || "";
    const baseNote = profile.id === "colab"
      ? "启用仅监测已有 Colab 连接，不会新建或续租实例。"
      : "点击上方入口使用本地电脑的视频库。";
    text("#health-source-note", `${unavailable ? unavailable + " " : ""}${baseNote} 此处开关仅控制健康监测。`);
    const notice = $("#health-disabled-notice");
    notice.hidden = !disabled;
    notice.textContent = profile.stopping
      ? "监测已关闭，不再发起后续健康请求；已开始的检查正在结束。下方仅保留历史数据，不代表当前服务状态。"
      : "监测已停用，不会向此目标发送健康请求。历史记录仍然保留；停用不代表服务故障，也不会停止推理服务。";
    const archiveKey = `${profile.id}:${disabled}`;
    if (archiveState !== archiveKey) {
      $("#health-archive").open = false;
      archiveState = archiveKey;
    }
    text("#health-archive-label", disabled ? "展开已保存的历史记录（不代表当前健康状态）" : "查看最近 3 小时的观测详情");
    text("#health-freshness-label", disabled ? "最后探测（历史）" : "最后探测新鲜度");
    text("#health-availability-label", disabled ? "探测样本可用率（历史）" : "探测样本可用率");
    text("#health-latency-label", disabled ? "就绪探测 P95（历史）" : "就绪探测 P95");
  }

  // JSON 仅投影展示字段，不暴露密钥、配置路径或未来新增的内部字段。
  function displayJSON(summary, entries) {
    const fields = ["window_hours", "since", "until", "observed_since", "observed_until",
      "latest_status", "last_probe_at", "last_probe_age_s", "stale", "observed_count", "success_count",
      "failure_count", "busy_count", "deep_success_count", "availability_pct", "latency_p50_ms", "latency_p95_ms",
      "last_inference_success_at", "consecutive_successes", "recovery_threshold", "consecutive_failures", "failure_threshold",
      "boot_id", "model", "model_revision", "lease_expires_at", "monitor_running", "status_counts"];
    const projected = Object.fromEntries(fields.filter(key => Object.hasOwn(summary, key)).map(key => [key, summary[key]]));
    projected.profile = { id: summary.profile.id, enabled: summary.profile.enabled };
    projected.entries = entries.map(entry => ({
      updated_at: entry.updated_at, status: entry.status, outcome: entry.outcome, reason: entry.reason,
      last_inference_success_at: entry.last_inference_success_at,
      checks: Object.fromEntries(["live", "ready", "inference"].filter(key => entry.checks?.[key]).map(key => [key, {
        http_status: entry.checks[key].http_status, latency_ms: entry.checks[key].latency_ms,
      }])),
    }));
    return JSON.stringify(projected, null, 2);
  }

  function render(summary, secondsSinceRead = 0) {
    renderProfile(summary);
    const profile = summary.profile;
    const disabled = profile.enabled !== true;
    const entries = Array.isArray(summary.entries) ? summary.entries.filter(entry => entry && typeof entry === "object") : [];
    const observed = count(summary.observed_count);
    const lastProbe = summary.last_probe_at;
    const timestampAge = lastProbe && !Number.isNaN(Date.parse(lastProbe)) ? Math.max(0, (Date.now() - Date.parse(lastProbe)) / 1000) : null;
    const baseAge = numeric(summary.last_probe_age_s);
    const age = baseAge == null ? timestampAge : baseAge + secondsSinceRead;
    const stale = observed > 0 && (summary.stale === true || age == null || age > 180);
    const lastStatus = String(summary.latest_status || "UNKNOWN").toUpperCase();
    const terminal = ["EXPIRED", "STOPPED", "NOT_CONFIGURED", "RELEASE_UNCONFIRMED", "DISABLED"].includes(lastStatus);
    const currentStatus = disabled ? "DISABLED" : stale && !terminal ? "STALE" : lastStatus;
    const showStatus = disabled || observed > 0 || terminal || ["MONITOR_ERROR", "CONFIG_ERROR"].includes(lastStatus);
    text("#health-status", showStatus ? (statusLabels[currentStatus] || currentStatus) : "尚无探测记录");
    $("#health-status").className = "health-status" + (!disabled && showStatus ? " " + category(currentStatus) : "");
    const monitorState = summary.monitor_running === true ? "监测器运行中" : summary.monitor_running === false ? "监测器未运行" : "监测器状态未知";
    const recoverySuccesses = numeric(summary.consecutive_successes);
    const recoveryThreshold = numeric(summary.recovery_threshold);
    const recoveryNote = currentStatus === "RECOVERING" && !disabled && !stale
      ? recoverySuccesses != null && recoveryThreshold > 0
        ? `本轮检查通过 · 连续成功 ${Math.min(count(recoverySuccesses), count(recoveryThreshold))}/${count(recoveryThreshold)} 轮，等待恢复确认`
        : "本轮检查通过 · 故障或实例变化后，等待连续成功确认"
      : null;
    const statusNote = recoveryNote ? `${recoveryNote} · ${monitorState}`
      : terminal ? monitorState
        : observed ? `${stale ? "最后结果：" + (statusLabels[lastStatus] || lastStatus) + " · " : ""}${monitorState}`
          : `${monitorState} · 等待首轮探测`;
    text("#health-status-note", disabled ? profile.stopping ? "不再发起新请求 · 等待在途检查结束" : "不发送健康请求 · 历史已保留" : statusNote);
    text("#health-freshness", disabled ? "已暂停" : observed && age != null ? ageLabel(age) : "—");
    text("#health-last-probe", disabled ? lastProbe ? `最近记录：${localTime(lastProbe)}（历史）` : "尚无历史记录" : lastProbe ? localTime(lastProbe) + (stale ? " · 已超过新鲜度阈值" : " · 180 秒内有效") : "超过 180 秒未采样不能视为健康");
    const availability = numeric(summary.availability_pct);
    text("#health-availability", observed && availability != null ? `${availability.toFixed(1)}%` : "—");
    text("#health-sample-counts", observed ? `成功 ${count(summary.success_count)} / 共 ${observed} 轮 · 失败 ${count(summary.failure_count)} · 忙碌 ${count(summary.busy_count)}` : "无观测样本，无法计算可用率");
    text("#health-p95", milliseconds(summary.latency_p95_ms));
    text("#health-p50", `包含网络耗时 · P50：${milliseconds(summary.latency_p50_ms)}`);
    text("#health-window", `最近 3 小时滚动窗口：${localTime(summary.since)} 至 ${localTime(summary.until)} · 实际观测 ${observed} 轮`);
    const timestamps = entries.map(entry => entry.updated_at).filter(value => value && !Number.isNaN(Date.parse(value))).sort((a, b) => Date.parse(a) - Date.parse(b));
    const coverageStart = summary.observed_since || timestamps[0];
    const coverageEnd = summary.observed_until || timestamps[timestamps.length - 1];
    const coverageLabel = summary.observed_since ? "实际观测覆盖" : "当前可见记录覆盖";
    text("#health-coverage", observed && coverageStart ? `${coverageLabel}：${localTime(coverageStart)} 至 ${localTime(coverageEnd)}；尚未覆盖完整窗口时不补造历史。` : "尚无探测记录；从首轮真实采样开始累计历史。");
    text("#health-last-inference", summary.last_inference_success_at ? localTime(summary.last_inference_success_at) : "尚无成功的真实推理检查");
    text("#health-expiry-label", profile.lease_managed ? "服务到期时间" : "服务运行方式");
    text("#health-expires", profile.lease_managed ? summary.lease_expires_at ? localTime(summary.lease_expires_at) : "尚无有效租约信息" : "常驻服务（无 Colab 租约限制）");
    text("#health-model", summary.model || "—");
    text("#health-revision", summary.model_revision || "—");
    text("#health-boot", summary.boot_id || "—");
    text("#health-deep-count", `${count(summary.deep_success_count)} 次`);
    text("#health-log-count", `显示 ${entries.length} 条，最近在前，最多 180 条`);
    $("#health-log-rows").innerHTML = entries.map(entry => {
      const reason = reasonLabels[entry.reason] || entry.reason || "—";
      return `<tr><td>${esc(localTime(entry.updated_at))}</td><td>${tableStatus(entry.status)}</td><td>${esc(reason)}</td>${checkCell(entry.checks?.live)}${checkCell(entry.checks?.ready)}${checkCell(entry.checks?.inference)}</tr>`;
    }).join("") || '<tr><td colspan="6" class="mute">尚无探测记录</td></tr>';
    text("#health-json", displayJSON(summary, entries));
    syncControls();
  }

  function acceptSummary(summary, expectedProfile = null) {
    if (!summary || !validProfile(summary.profile?.id) || (expectedProfile && summary.profile.id !== expectedProfile)) {
      throw new Error("健康监测配置响应不完整");
    }
    selectedProfile = summary.profile.id;
    lastSummary = summary;
    lastReadAt = performance.now();
    render(summary);
    $("#health-error").hidden = true;
    text("#health-refresh-state", `页面更新于 ${localTime(new Date().toISOString())}`);
  }

  function schedule() {
    clearTimeout(timer);
    timer = visible() && !mutationBusy ? setTimeout(loadHealth, 30000) : null;
  }

  function abortRead() {
    activeRequest?.abort();
    activeRequest = null;
    requestNumber += 1;
  }

  async function loadHealth() {
    if (!visible() || activeRequest || mutationBusy) return;
    const controller = new AbortController();
    activeRequest = controller;
    const number = ++requestNumber;
    const requestedProfile = readSavedSelection ? null : selectedProfile;
    syncControls();
    text("#health-refresh-state", "读取历史记录中…");
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 15000);
    try {
      const query = new URLSearchParams({ hours: "3", limit: "180" });
      if (requestedProfile) query.set("profile", requestedProfile);
      const summary = await api(`/api/service-health?${query}`, { signal: controller.signal, cache: "no-store" });
      if (number !== requestNumber || !visible()) return;
      acceptSummary(summary, requestedProfile);
      readSavedSelection = false;
    } catch (error) {
      if (number !== requestNumber || !visible()) return;
      if (error.name === "AbortError" && !timedOut) return;
      if (lastSummary && lastReadAt != null) render(lastSummary, (performance.now() - lastReadAt) / 1000);
      const notice = $("#health-error");
      notice.textContent = timedOut ? "读取记录超时。保留上次展示的数据，请稍后刷新。" : "读取健康记录失败。保留上次展示的数据，请确认 Web 服务已更新后刷新。";
      notice.hidden = false;
      text("#health-refresh-state", "刷新失败");
    } finally {
      clearTimeout(timeout);
      if (activeRequest === controller) {
        activeRequest = null;
        syncControls();
        schedule();
      }
    }
  }

  async function updateProfile(id, changes) {
    if (mutationBusy || !validProfile(id)) return;
    abortRead();
    clearTimeout(timer);
    mutationBusy = true;
    syncControls();
    text("#health-refresh-state", changes.selected ? "切换监测目标中…" : "保存监测开关中…");
    const controller = new AbortController();
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 15000);
    try {
      // 修改请求串行完成；不会因快速切换或页面隐藏而重复修改另一个目标。
      const summary = await api(`/api/service-health/profiles/${id}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(changes), signal: controller.signal,
      });
      acceptSummary(summary, id);
      readSavedSelection = false;
    } catch (error) {
      if (lastSummary && lastReadAt != null) render(lastSummary, (performance.now() - lastReadAt) / 1000);
      const notice = $("#health-error");
      notice.textContent = timedOut
        ? "保存超时，已保留上次画面。请刷新核对服务器保存的监测设置。"
        : `未完成设置，保留上次画面。${error.name === "AbortError" ? "请刷新后重试。" : error.message}`;
      notice.hidden = false;
      text("#health-refresh-state", "设置未确认");
      readSavedSelection = true;
    } finally {
      clearTimeout(timeout);
      mutationBusy = false;
      syncControls();
      schedule();
    }
  }

  function deactivate() {
    clearTimeout(timer);
    timer = null;
    abortRead();
    syncControls();
  }

  profileButtons.forEach((button, index) => {
    button.onclick = () => {
      const id = button.dataset.healthProfile;
      if (id !== selectedProfile) updateProfile(id, { selected: true });
    };
    button.onkeydown = event => {
      const destinations = { ArrowLeft: (index + profileButtons.length - 1) % profileButtons.length, ArrowRight: (index + 1) % profileButtons.length, Home: 0, End: profileButtons.length - 1 };
      if (!Object.hasOwn(destinations, event.key) || mutationBusy) return;
      event.preventDefault();
      const next = profileButtons[destinations[event.key]];
      next.focus();
      next.click();
    };
  });
  toggleButton.onclick = () => {
    const profile = currentProfile();
    if (profile) updateProfile(profile.id, { enabled: !profile.enabled });
  };
  refreshButton.onclick = loadHealth;
  TABS.health = loadHealth;
  new MutationObserver(() => {
    if (!visible()) deactivate();
    else if (!activeRequest && !mutationBusy) schedule();
  }).observe(panel, { attributes: true, attributeFilter: ["hidden"] });
  document.addEventListener("visibilitychange", () => {
    if (visible()) loadHealth();
    else deactivate();
  });
})();
