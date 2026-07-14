const STORAGE_KEY = "academic-rag.sessions.v3";

const state = {
  sessions: [],
  activeSessionId: null,
  activeSources: [],
  jobs: new Map(),
  documents: [],
  selectedSources: [],
  currentView: "chat",
  evidenceVisible: window.innerWidth > 820,
  previewDocument: null,
  libraryQuery: "",
  librarySort: "recent",
  scopePopoverOpen: false,
};

const els = {
  navItems: [...document.querySelectorAll("[data-view]")],
  viewPanels: [...document.querySelectorAll("[data-view-panel]")],
  chatLayout: document.querySelector("#chatLayout"),
  healthText: document.querySelector("#healthText"),
  statusDot: document.querySelector("#statusDot"),
  libraryNavCount: document.querySelector("#libraryNavCount"),
  refreshDocumentsBtn: document.querySelector("#refreshDocumentsBtn"),
  pdfInput: document.querySelector("#pdfInput"),
  jobList: document.querySelector("#jobList"),
  documentList: document.querySelector("#documentList"),
  librarySearchInput: document.querySelector("#librarySearchInput"),
  librarySort: document.querySelector("#librarySort"),
  documentStat: document.querySelector("#documentStat"),
  chunkStat: document.querySelector("#chunkStat"),
  selectedStat: document.querySelector("#selectedStat"),
  selectionBar: document.querySelector("#selectionBar"),
  selectionCount: document.querySelector("#selectionCount"),
  clearSelectionBtn: document.querySelector("#clearSelectionBtn"),
  askSelectionBtn: document.querySelector("#askSelectionBtn"),
  sessionList: document.querySelector("#sessionList"),
  newSessionBtn: document.querySelector("#newSessionBtn"),
  clearCurrentChatBtn: document.querySelector("#clearCurrentChatBtn"),
  clearAllChatsBtn: document.querySelector("#clearAllChatsBtn"),
  activeSessionTitle: document.querySelector("#activeSessionTitle"),
  activeSessionMeta: document.querySelector("#activeSessionMeta"),
  agentToggle: document.querySelector("#agentToggle"),
  scopeBar: document.querySelector("#scopeBar"),
  scopeChips: document.querySelector("#scopeChips"),
  clearScopeBtn: document.querySelector("#clearScopeBtn"),
  messages: document.querySelector("#messages"),
  chatForm: document.querySelector("#chatForm"),
  questionInput: document.querySelector("#questionInput"),
  memoryToggle: document.querySelector("#memoryToggle"),
  composerLibraryBtn: document.querySelector("#composerLibraryBtn"),
  selectedPaperSummary: document.querySelector("#selectedPaperSummary"),
  sendBtn: document.querySelector("#sendBtn"),
  toggleEvidenceBtn: document.querySelector("#toggleEvidenceBtn"),
  closeEvidenceBtn: document.querySelector("#closeEvidenceBtn"),
  searchForm: document.querySelector("#searchForm"),
  searchInput: document.querySelector("#searchInput"),
  sourceList: document.querySelector("#sourceList"),
  sourceCount: document.querySelector("#sourceCount"),
  previewDrawer: document.querySelector("#previewDrawer"),
  drawerBackdrop: document.querySelector("#drawerBackdrop"),
  previewType: document.querySelector("#previewType"),
  previewMeta: document.querySelector("#previewMeta"),
  pdfTitle: document.querySelector("#pdfTitle"),
  pdfPreviewContent: document.querySelector("#pdfPreviewContent"),
  pdfOpenLink: document.querySelector("#pdfOpenLink"),
  closePdfBtn: document.querySelector("#closePdfBtn"),
  scopePreviewBtn: document.querySelector("#scopePreviewBtn"),
  toastRegion: document.querySelector("#toastRegion"),
};

function uid(prefix) {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  els.toastRegion.appendChild(toast);
  window.setTimeout(() => toast.remove(), 3600);
}

function formatDate(value, detailed = false) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("zh-CN", detailed
    ? { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }
    : { month: "2-digit", day: "2-digit" }
  ).format(date);
}

function switchView(view) {
  state.currentView = view;
  for (const item of els.navItems) item.classList.toggle("active", item.dataset.view === view);
  for (const panel of els.viewPanels) panel.classList.toggle("active", panel.dataset.viewPanel === view);
  if (view === "library") renderDocuments();
  if (view === "chat") window.setTimeout(() => els.questionInput.focus(), 80);
}

function loadSessions() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    state.sessions = Array.isArray(parsed) ? parsed : [];
  } catch {
    state.sessions = [];
  }
  if (state.sessions.length === 0) {
    createSession(false);
    return;
  }
  state.activeSessionId = state.sessions[0].id;
  const lastAssistant = [...state.sessions[0].messages].reverse().find((item) => item.role === "assistant");
  state.activeSources = lastAssistant?.sources || [];
}

function saveSessions() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions));
}

function createSession(render = true) {
  const now = new Date().toISOString();
  const session = { id: uid("chat"), title: "新对话", messages: [], createdAt: now, updatedAt: now };
  state.sessions.unshift(session);
  state.activeSessionId = session.id;
  state.activeSources = [];
  saveSessions();
  switchView("chat");
  if (render) renderAll();
  return session;
}

function activeSession() {
  return state.sessions.find((session) => session.id === state.activeSessionId);
}

function updateSessionTitle(session, question) {
  if (!session || !["新对话", "New chat"].includes(session.title)) return;
  session.title = question.trim().slice(0, 26) || "新对话";
}

function renderAll() {
  renderSessions();
  renderMessages();
  renderSources(state.activeSources);
  renderScope();
  renderLibraryStats();
}

function renderSessions() {
  els.sessionList.innerHTML = "";
  for (const session of state.sessions) {
    const button = document.createElement("button");
    button.className = `session-item ${session.id === state.activeSessionId ? "active" : ""}`;
    button.type = "button";
    button.title = session.title;
    button.innerHTML = `<strong>${escapeHtml(session.title)}</strong><span class="meta-line">${session.messages.length} 条消息</span>`;
    button.addEventListener("click", () => {
      state.activeSessionId = session.id;
      const lastAssistant = [...session.messages].reverse().find((item) => item.role === "assistant");
      state.activeSources = lastAssistant?.sources || [];
      switchView("chat");
      renderAll();
    });
    els.sessionList.appendChild(button);
  }
}

function emptyChatMarkup() {
  return `
    <div class="empty-state">
      <div class="empty-mark"><svg><use href="#icon-sparkle"></use></svg></div>
      <h2>今天想研究什么？</h2>
      <p>从文件库选择论文限定范围，或直接在全部文献中检索。每个回答都会保留可核对的证据片段。</p>
      <div class="prompt-suggestions">
        <button class="prompt-suggestion" type="button">概括最近上传论文的核心贡献</button>
        <button class="prompt-suggestion" type="button">比较两篇论文的方法与实验结论</button>
        <button class="prompt-suggestion" type="button">检索关于 Non-IID 收敛性的论述</button>
        <button class="prompt-suggestion" type="button">找出方法的关键假设和局限性</button>
      </div>
    </div>`;
}

function renderMessages() {
  const session = activeSession();
  els.activeSessionTitle.textContent = session?.title || "新对话";
  const scopeCount = state.selectedSources.length;
  els.activeSessionMeta.textContent = scopeCount
    ? `限定检索 ${scopeCount} 篇论文 · ${session?.messages.length || 0} 条消息`
    : `全部论文 · ${session?.messages.length || 0} 条消息`;
  els.messages.innerHTML = "";

  if (!session || session.messages.length === 0) {
    els.messages.innerHTML = emptyChatMarkup();
    for (const button of els.messages.querySelectorAll(".prompt-suggestion")) {
      button.addEventListener("click", () => {
        els.questionInput.value = button.textContent.trim();
        autoResizeComposer();
        els.questionInput.focus();
      });
    }
    return;
  }

  for (const message of session.messages) {
    const item = document.createElement("article");
    item.className = `message ${message.role}`;
    if (message.role === "assistant") {
      const avatar = document.createElement("div");
      avatar.className = "message-avatar";
      avatar.innerHTML = `<svg><use href="#icon-sparkle"></use></svg>`;
      item.appendChild(avatar);
    }
    const bubble = document.createElement("div");
    bubble.className = `bubble ${message.pending ? "pending" : ""}`;
    bubble.textContent = message.content || "";
    item.appendChild(bubble);
    els.messages.appendChild(item);
  }
  els.messages.scrollTop = els.messages.scrollHeight;
}

function documentLabel(sourceName) {
  const doc = state.documents.find((item) => item.source_name === sourceName);
  return doc?.paper_title || doc?.source_name || sourceName;
}

function renderScope() {
  const selectedCount = state.selectedSources.length;
  els.scopeBar.classList.toggle("visible", selectedCount > 0);
  if (selectedCount === 0) {
    state.scopePopoverOpen = false;
    els.scopeChips.innerHTML = "";
  } else {
    const firstTitle = documentLabel(state.selectedSources[0]);
    const summary = selectedCount === 1 ? firstTitle : `${firstTitle} 等 ${selectedCount} 篇论文`;
    const items = state.selectedSources.map((source) => `
      <div class="scope-popover-item">
        <span title="${escapeHtml(documentLabel(source))}">${escapeHtml(documentLabel(source))}</span>
        <button class="scope-popover-remove" type="button" data-remove-source="${escapeHtml(source)}" title="移出检索范围">
          <svg><use href="#icon-close"></use></svg>
        </button>
      </div>`).join("");
    els.scopeChips.innerHTML = `
      <button class="scope-summary-button" type="button" aria-expanded="${state.scopePopoverOpen}">
        <span>${escapeHtml(summary)}</span>
        <svg><use href="#icon-arrow"></use></svg>
      </button>
      <div class="scope-popover ${state.scopePopoverOpen ? "visible" : ""}">
        <div class="scope-popover-header"><strong>当前检索范围</strong><span>${selectedCount} 篇论文</span></div>
        <div class="scope-popover-list">${items}</div>
      </div>`;
    els.scopeChips.querySelector(".scope-summary-button").addEventListener("click", (event) => {
      event.stopPropagation();
      state.scopePopoverOpen = !state.scopePopoverOpen;
      renderScope();
    });
    for (const button of els.scopeChips.querySelectorAll("[data-remove-source]")) {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        const source = button.dataset.removeSource;
        state.selectedSources = state.selectedSources.filter((item) => item !== source);
        state.scopePopoverOpen = state.selectedSources.length > 0;
        renderDocuments();
        renderScope();
      });
    }
  }
  if (selectedCount === 0) {
    els.selectedPaperSummary.textContent = "选择论文";
    els.composerLibraryBtn.title = "从文件库选择论文并限定检索范围";
  } else {
    const firstTitle = documentLabel(state.selectedSources[0]);
    els.selectedPaperSummary.textContent = selectedCount === 1
      ? `${firstTitle} · 1 篇论文`
      : `${firstTitle} 等 ${selectedCount} 篇论文`;
    els.composerLibraryBtn.title = `当前限定 ${selectedCount} 篇论文，点击修改`;
  }
  els.composerLibraryBtn.classList.toggle("has-selection", selectedCount > 0);
  renderLibraryStats();
}

function sortedFilteredDocuments() {
  const query = state.libraryQuery.trim().toLocaleLowerCase("zh-CN");
  const result = state.documents.filter((doc) => {
    if (!query) return true;
    return `${doc.paper_title || ""} ${doc.source_name || ""}`.toLocaleLowerCase("zh-CN").includes(query);
  });
  result.sort((a, b) => {
    if (state.librarySort === "name") return documentLabel(a.source_name).localeCompare(documentLabel(b.source_name), "zh-CN");
    if (state.librarySort === "chunks") return Number(b.chunk_count || 0) - Number(a.chunk_count || 0);
    return new Date(b.created_at || 0) - new Date(a.created_at || 0);
  });
  return result;
}

function renderDocuments(documents) {
  if (Array.isArray(documents)) state.documents = documents;
  const visibleDocuments = sortedFilteredDocuments();
  els.documentList.innerHTML = "";

  if (visibleDocuments.length === 0) {
    const hasDocuments = state.documents.length > 0;
    els.documentList.innerHTML = `
      <div class="document-empty">
        <svg><use href="#${hasDocuments ? "icon-search" : "icon-library"}"></use></svg>
        <strong>${hasDocuments ? "没有匹配的论文" : "文件库还是空的"}</strong>
        <p>${hasDocuments ? "换一个标题或文件名试试" : "上传 PDF 后会自动解析并建立索引"}</p>
      </div>`;
    renderLibraryStats();
    return;
  }

  for (const doc of visibleDocuments) {
    const selected = state.selectedSources.includes(doc.source_name);
    const row = document.createElement("article");
    row.className = `document-row ${selected ? "selected" : ""}`;
    const title = doc.paper_title || doc.source_name || "未命名论文";
    row.innerHTML = `
      <button class="file-checkbox" type="button" aria-label="${selected ? "移出" : "加入"}检索范围" aria-pressed="${selected}">
        <svg><use href="#icon-check"></use></svg>
      </button>
      <div class="file-name-cell">
        <div class="file-icon"><span>PDF</span></div>
        <div class="file-title">
          <button type="button" title="${escapeHtml(title)}">${escapeHtml(title)}</button>
          <span>${escapeHtml(doc.source_name || "")}</span>
        </div>
      </div>
      <span class="index-badge ${doc.chunk_count ? "" : "missing"}">${doc.chunk_count ? "已索引" : "待索引"}</span>
      <span class="document-value">${Number(doc.chunk_count || 0).toLocaleString()}</span>
      <span class="document-value">${formatDate(doc.created_at)}</span>
      <button class="row-action" type="button" title="预览论文"><svg><use href="#icon-arrow"></use></svg></button>`;
    const checkbox = row.querySelector(".file-checkbox");
    const titleButton = row.querySelector(".file-title button");
    const actionButton = row.querySelector(".row-action");
    checkbox.addEventListener("click", () => toggleDocumentScope(doc));
    titleButton.addEventListener("click", () => openDocumentPreview(doc));
    actionButton.addEventListener("click", () => openDocumentPreview(doc));
    els.documentList.appendChild(row);
  }
  renderLibraryStats();
}

function renderLibraryStats() {
  const chunks = state.documents.reduce((sum, doc) => sum + Number(doc.chunk_count || 0), 0);
  els.libraryNavCount.textContent = String(state.documents.length);
  els.documentStat.textContent = state.documents.length.toLocaleString();
  els.chunkStat.textContent = chunks.toLocaleString();
  els.selectedStat.textContent = state.selectedSources.length.toLocaleString();
  els.selectionCount.textContent = String(state.selectedSources.length);
  els.selectionBar.classList.toggle("visible", state.selectedSources.length > 0 && state.currentView === "library");
}

function toggleDocumentScope(doc) {
  const index = state.selectedSources.indexOf(doc.source_name);
  if (index >= 0) state.selectedSources.splice(index, 1);
  else state.selectedSources.push(doc.source_name);
  renderDocuments();
  renderScope();
  if (state.previewDocument?.id === doc.id) updatePreviewScopeButton();
}

function updatePreviewScopeButton() {
  const doc = state.previewDocument;
  if (!doc) return;
  const selected = state.selectedSources.includes(doc.source_name);
  els.scopePreviewBtn.textContent = selected ? "移出检索范围" : "加入检索范围";
}

async function openDocumentPreview(doc) {
  state.previewDocument = doc;
  els.previewDrawer.classList.add("visible");
  els.drawerBackdrop.classList.add("visible");
  els.previewDrawer.setAttribute("aria-hidden", "false");
  els.previewType.textContent = "PAPER PREVIEW";
  els.pdfTitle.textContent = doc.paper_title || doc.source_name || "论文预览";
  els.previewMeta.innerHTML = `<span>${Number(doc.chunk_count || 0)} 个切片</span><span>${doc.has_pdf ? "原文可用" : "仅索引内容"}</span><span>${formatDate(doc.created_at, true)}</span>`;
  els.pdfPreviewContent.textContent = "正在提取摘要与预览…";
  els.pdfOpenLink.href = doc.pdf_url || "#";
  els.pdfOpenLink.style.display = doc.pdf_url ? "inline-flex" : "none";
  updatePreviewScopeButton();

  try {
    const response = await fetch(`/documents/${doc.id}/preview`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "无法加载预览");
    els.previewType.textContent = payload.preview_type === "abstract" ? "ABSTRACT" : "PAPER PREVIEW";
    els.pdfOpenLink.href = payload.pdf_url || doc.pdf_url || "#";
    els.pdfOpenLink.style.display = payload.pdf_url || doc.pdf_url ? "inline-flex" : "none";
    els.pdfPreviewContent.textContent = payload.preview_text || "当前论文没有可用的文本预览，请打开原文查看。";
  } catch (error) {
    els.pdfPreviewContent.textContent = `预览加载失败：${error.message}`;
  }
}

function closePdf() {
  state.previewDocument = null;
  els.previewDrawer.classList.remove("visible");
  els.drawerBackdrop.classList.remove("visible");
  els.previewDrawer.setAttribute("aria-hidden", "true");
}

function renderJobs() {
  els.jobList.innerHTML = "";
  for (const job of state.jobs.values()) {
    const card = document.createElement("article");
    card.className = "job-card";
    const labels = { queued: "等待索引", started: "正在解析", finished: "索引完成", failed: "索引失败" };
    card.innerHTML = `
      <div class="file-icon"><span>PDF</span></div>
      <strong>${escapeHtml(job.filename || "索引任务")}</strong>
      <div class="status ${escapeHtml(job.status || "queued")}">${labels[job.status] || escapeHtml(job.status || "等待索引")}</div>
      <div class="meta-line">${job.chunks_added ? `${job.chunks_added} 个切片` : "后台任务"}</div>`;
    els.jobList.appendChild(card);
  }
}

function normalizeSources(sources = []) {
  return sources.map((source) => ({
    source: source.source || source.metadata?.source || "未知来源",
    page: source.page ?? source.metadata?.page ?? null,
    score: source.score,
    chunk_id: source.chunk_id || source.id,
    content_preview: source.content_preview || source.text || source.content || "",
  }));
}

function renderSources(sources = []) {
  state.activeSources = normalizeSources(sources);
  els.sourceCount.textContent = String(state.activeSources.length);
  els.sourceList.innerHTML = "";
  if (state.activeSources.length === 0) {
    els.sourceList.innerHTML = `
      <div class="source-card empty-card">
        <strong>还没有检索证据</strong>
        <p>提出问题后，相关论文片段及页码会展示在这里。</p>
      </div>`;
    return;
  }

  for (const source of state.activeSources) {
    const card = document.createElement("article");
    card.className = "source-card";
    const page = source.page ? `第 ${source.page} 页` : "页码未知";
    const score = typeof source.score === "number" ? source.score.toFixed(4) : source.score;
    card.innerHTML = `
      <strong title="${escapeHtml(source.source)}">${escapeHtml(documentLabel(source.source))}</strong>
      <div class="meta-line">${page} · 相关度 ${escapeHtml(String(score || "—"))}</div>
      <p>${escapeHtml(source.content_preview)}</p>`;
    card.addEventListener("click", () => {
      const doc = state.documents.find((item) => item.source_name === source.source);
      if (doc) openDocumentPreview(doc);
    });
    els.sourceList.appendChild(card);
  }
}

async function refreshHealth() {
  els.statusDot.className = "status-dot";
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("health failed");
    const payload = await response.json();
    els.healthText.textContent = `${Number(payload.index_size || 0).toLocaleString()} 个向量就绪`;
    els.statusDot.className = "status-dot online";
  } catch {
    els.healthText.textContent = "服务未连接";
    els.statusDot.className = "status-dot offline";
  }
}

async function refreshDocuments() {
  try {
    const response = await fetch("/documents");
    if (!response.ok) throw new Error("无法获取文件列表");
    const payload = await response.json();
    renderDocuments(payload.documents || []);
  } catch (error) {
    renderDocuments([]);
    showToast(error.message, "error");
  }
}

async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  showToast(`正在上传 ${file.name}`);
  const response = await fetch("/upload", { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "上传失败");
  state.jobs.set(payload.job_id, payload);
  renderJobs();
  showToast("上传完成，已进入后台索引队列");
  pollJob(payload.job_id);
}

async function pollJob(jobId) {
  const timer = window.setInterval(async () => {
    try {
      const response = await fetch(`/jobs/${jobId}`);
      const payload = await response.json();
      state.jobs.set(jobId, payload);
      renderJobs();
      if (["finished", "failed", "stopped", "canceled"].includes(payload.status)) {
        window.clearInterval(timer);
        if (payload.status === "finished") showToast(`${payload.filename || "PDF"} 已完成索引`);
        else showToast(`${payload.filename || "PDF"} 索引失败`, "error");
        await Promise.all([refreshHealth(), refreshDocuments()]);
      }
    } catch {
      window.clearInterval(timer);
    }
  }, 1800);
}

async function sendQuestion(question) {
  let session = activeSession();
  if (!session) session = createSession(false);
  updateSessionTitle(session, question);
  session.messages.push({ role: "user", content: question });
  const assistantMessage = { role: "assistant", content: "", sources: [], pending: true };
  session.messages.push(assistantMessage);
  session.updatedAt = new Date().toISOString();
  saveSessions();
  renderAll();
  els.sendBtn.disabled = true;

  try {
    await streamQuery(question, session.id, assistantMessage);
    if (assistantMessage.sources.length === 0) {
      assistantMessage.sources = await fetchSourcesFallback(question);
      renderSources(assistantMessage.sources);
    }
  } catch (error) {
    assistantMessage.content = `请求失败：${error.message}`;
  } finally {
    assistantMessage.pending = false;
    session.updatedAt = new Date().toISOString();
    saveSessions();
    els.sendBtn.disabled = false;
    renderAll();
  }
}

async function streamQuery(question, sessionId, assistantMessage) {
  const response = await fetch("/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      use_agent: els.agentToggle.checked,
      use_memory: els.memoryToggle.checked,
      source_names: state.selectedSources,
    }),
  });
  if (!response.ok || !response.body) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "问答接口不可用");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || "";
    for (const event of events) handleSseEvent(event, assistantMessage);
  }
  if (buffer.trim()) handleSseEvent(buffer, assistantMessage);
}

function handleSseEvent(rawEvent, assistantMessage) {
  const dataLines = rawEvent.split(/\r?\n/).filter((line) => line.startsWith("data: ")).map((line) => line.slice(6));
  if (dataLines.length === 0) return;
  const payload = JSON.parse(dataLines.join("\n"));
  if (payload.type === "sources") {
    assistantMessage.sources = normalizeSources(payload.data || []);
    renderSources(assistantMessage.sources);
  }
  if (payload.type === "token") {
    assistantMessage.content += payload.data || "";
    renderMessages();
  }
}

async function fetchSourcesFallback(query) {
  const response = await fetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 8, score_threshold: 0, source_names: state.selectedSources }),
  });
  const payload = await response.json();
  if (!response.ok) return [];
  return normalizeSources(payload.retrieved_chunks || []);
}

async function searchOnly(query) {
  const sources = await fetchSourcesFallback(query);
  renderSources(sources);
  state.evidenceVisible = true;
  updateEvidencePanel();
}

async function clearBackendMemory(sessionId) {
  const response = await fetch("/memory/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || "无法清除后端记忆");
}

async function clearCurrentChat() {
  const session = activeSession();
  if (!session) return;
  await clearBackendMemory(session.id);
  session.messages = [];
  session.updatedAt = new Date().toISOString();
  state.activeSources = [];
  saveSessions();
  renderAll();
}

async function clearAllChats() {
  await clearBackendMemory(null);
  state.sessions = [];
  createSession(false);
  saveSessions();
  renderAll();
}

function autoResizeComposer() {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height = `${Math.min(els.questionInput.scrollHeight, 180)}px`;
}

function updateEvidencePanel() {
  els.chatLayout.classList.toggle("evidence-hidden", !state.evidenceVisible);
}

for (const navItem of els.navItems) navItem.addEventListener("click", () => switchView(navItem.dataset.view));
els.composerLibraryBtn.addEventListener("click", () => switchView("library"));
els.newSessionBtn.addEventListener("click", () => createSession(true));

els.refreshDocumentsBtn.addEventListener("click", async () => {
  await Promise.all([refreshHealth(), refreshDocuments()]);
  showToast("文件库已刷新");
});

els.pdfInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try { await uploadPdf(file); }
  catch (error) { showToast(error.message, "error"); }
  finally { els.pdfInput.value = ""; }
});

els.librarySearchInput.addEventListener("input", () => {
  state.libraryQuery = els.librarySearchInput.value;
  renderDocuments();
});
els.librarySort.addEventListener("change", () => {
  state.librarySort = els.librarySort.value;
  renderDocuments();
});
els.clearSelectionBtn.addEventListener("click", () => {
  state.selectedSources = [];
  renderDocuments();
  renderScope();
});
els.askSelectionBtn.addEventListener("click", () => switchView("chat"));
els.clearScopeBtn.addEventListener("click", () => {
  state.selectedSources = [];
  renderDocuments();
  renderScope();
});

els.closePdfBtn.addEventListener("click", closePdf);
els.drawerBackdrop.addEventListener("click", closePdf);
els.scopePreviewBtn.addEventListener("click", () => {
  if (state.previewDocument) toggleDocumentScope(state.previewDocument);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closePdf();
    if (state.scopePopoverOpen) {
      state.scopePopoverOpen = false;
      renderScope();
    }
  }
});
document.addEventListener("click", (event) => {
  if (!state.scopePopoverOpen || els.scopeChips.contains(event.target)) return;
  state.scopePopoverOpen = false;
  renderScope();
});

els.toggleEvidenceBtn.addEventListener("click", () => {
  state.evidenceVisible = !state.evidenceVisible;
  updateEvidencePanel();
});
els.closeEvidenceBtn.addEventListener("click", () => {
  state.evidenceVisible = false;
  updateEvidencePanel();
});

els.clearCurrentChatBtn.addEventListener("click", async () => {
  try { await clearCurrentChat(); showToast("当前对话已清空"); }
  catch (error) { showToast(error.message, "error"); }
});
els.clearAllChatsBtn.addEventListener("click", async () => {
  if (!window.confirm("清除全部本地对话和后端会话记忆？")) return;
  try { await clearAllChats(); showToast("全部对话记录已清除"); }
  catch (error) { showToast(error.message, "error"); }
});

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = els.questionInput.value.trim();
  if (!question || els.sendBtn.disabled) return;
  els.questionInput.value = "";
  autoResizeComposer();
  await sendQuestion(question);
});
els.questionInput.addEventListener("input", autoResizeComposer);
els.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.chatForm.requestSubmit();
  }
});

els.searchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = els.searchInput.value.trim();
  if (!query) return;
  try { await searchOnly(query); }
  catch (error) { showToast(error.message, "error"); }
});

loadSessions();
renderAll();
updateEvidencePanel();
Promise.all([refreshHealth(), refreshDocuments()]);
