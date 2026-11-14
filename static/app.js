/**
 * Frontend Application Logic for the Standalone Local-First Godot 4
 * Agentic Coding & 2D/3D Asset-Creation Harness.
 */

const state = {
  ws: null,
  attachments: [],
  currentFileResPath: "res://scenes/main_3d.tscn",
  yieldedToGodot: false,
  telemetry: null,
};

// ============================================================================
// Initialization & WebSocket Live Telemetry Stream
// ============================================================================
window.addEventListener("DOMContentLoaded", async () => {
  initWebSocket();
  bindUIEvents();
  await Promise.all([
    fetchStatus(),
    loadWorkspaceTree(),
    inspectWorkspaceFile("res://scenes/main_3d.tscn"),
    loadDiffHistory(),
  ]);
});

function initWebSocket() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${window.location.host}/ws`;
  const ws = new WebSocket(wsUrl);
  state.ws = ws;

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "telemetry_update") {
        updateTelemetryUI(msg.data);
      } else if (msg.type === "step_start") {
        handleLiveStepStart(msg.data);
      } else if (msg.type === "tool_start") {
        handleLiveToolStart(msg.data);
      } else if (msg.type === "tool_complete") {
        handleLiveToolComplete(msg.data);
        loadWorkspaceTree();
        loadDiffHistory();
      } else if (msg.type === "download_progress") {
        handleDownloadProgress(msg.data);
      } else if (msg.type === "workspace_changed") {
        if (msg.data?.workspace_dir) updateWorkspaceBadge(msg.data.workspace_dir);
        loadWorkspaceTree();
        loadDiffHistory();
      }
    } catch (err) {
      console.error("WS parse error:", err);
    }
  };

  ws.onclose = () => {
    setTimeout(initWebSocket, 2500);
  };
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    updateTelemetryUI(data);
  } catch (err) {
    console.error("Failed to fetch status:", err);
  }
}

// ============================================================================
// Live VRAM & Engine State Telemetry Bar Updates
// ============================================================================
function updateTelemetryUI(data) {
  if (!data) return;
  state.telemetry = data;

  const badge = document.getElementById("state-badge");
  badge.textContent = data.state;
  badge.className = "state-badge";
  if (data.state === "LLM_ACTIVE") {
    badge.classList.add("state-llm-active");
  } else if (data.state === "HYBRID_RAM_OVERFLOW") {
    badge.classList.add("state-overflow");
  } else if (
    data.state === "SWAPPING_TO_IMAGEGEN" ||
    data.state === "IMAGEGEN_ACTIVE" ||
    data.state === "RESTORING_LLM"
  ) {
    badge.classList.add("state-imagegen");
  } else if (data.state === "GODOT_YIELDED") {
    badge.classList.add("state-yielded");
  }

  const kvBadge = document.getElementById("kv-tier-badge");
  const kvTier = data.kv_cache_tier || "KV: VRAM";
  if (kvBadge) {
    kvBadge.textContent = kvTier;
    kvBadge.className = "state-badge";
    if (kvTier === "KV: VRAM") {
      kvBadge.classList.add("state-llm-active");
    } else if (kvTier === "KV: RAM Spill") {
      kvBadge.classList.add("state-overflow");
    } else {
      kvBadge.classList.add("state-yielded");
    }
  }

  document.getElementById("telemetry-status-text").textContent =
    data.status_bar_text || "VRAM: 8.4/12GB | Engine: Ornith-9B [VRAM Fast] | KV: VRAM | Blender: Ready";

  if (data.gpu) {
    const usedGb = data.gpu.used_vram_gb ?? 8.4;
    const totalGb = data.gpu.total_vram_gb ?? 12.0;
    const pct = Math.min(100, Math.round((usedGb / Math.max(totalGb, 1)) * 100));
    document.getElementById("vram-meter-text").textContent = `${usedGb.toFixed(1)} / ${totalGb.toFixed(1)} GB`;
    document.getElementById("vram-meter-fill").style.width = `${pct}%`;
  }

  if (data.system_ram) {
    const rUsed = data.system_ram.used_gb ?? 8.0;
    const rTotal = data.system_ram.total_gb ?? 32.0;
    const rPct = Math.min(100, Math.round((rUsed / Math.max(rTotal, 1)) * 100));
    document.getElementById("ram-meter-text").textContent = `${rUsed.toFixed(1)} / ${rTotal.toFixed(1)} GB`;
    document.getElementById("ram-meter-fill").style.width = `${rPct}%`;
  }

  const profileSel = document.getElementById("profile-select");
  if (data.active_profile_id && profileSel.value !== data.active_profile_id) {
    profileSel.value = data.active_profile_id;
  }

  const ctxTokens = data.estimated_context_tokens || 120;
  const maxCtx = data.active_profile?.context_window || 32768;
  const maxK = Math.round(maxCtx / 1024);
  document.getElementById("ctx-token-badge").textContent =
    `~${ctxTokens.toLocaleString()} / ${maxK}k tok`;

  if (data.workspace_dir) {
    updateWorkspaceBadge(data.workspace_dir);
  }

  // Update State Machine Grid on Tab 3
  document.querySelectorAll("#state-machine-grid .state-node").forEach((node) => {
    node.classList.toggle("active", node.dataset.state === data.state);
  });

  // Update 3-Tier KV Cache Grid on Tab 3
  document.querySelectorAll("#kv-tier-grid .state-node").forEach((node) => {
    node.classList.toggle("active", node.dataset.kvTier === kvTier);
  });

  const diskCache = data.kv_hierarchy?.tier3_disk_cache;
  const diskSummaryEl = document.getElementById("kv-disk-summary");
  if (diskCache && diskSummaryEl) {
    diskSummaryEl.textContent =
      `NVMe Disk (./cache/kv_disk): ${diskCache.total_mb} MB / ${diskCache.max_cap_gb} GB LRU Cap (${diskCache.session_count} slots | key: ${diskCache.active_session_key})`;
  }

  if (data.llama_cmd_preview) {
    document.getElementById("llama-cmd-box").textContent = data.llama_cmd_preview;
  }

  if (Array.isArray(data.transitions)) {
    const logEl = document.getElementById("transition-log");
    logEl.innerHTML = data.transitions
      .slice()
      .reverse()
      .map(
        (t) => `
        <div class="transition-entry">
          <div><strong>[${t.timestamp}]</strong> ${t.from_state || "INIT"} &rarr; <span style="color:#34d399">${t.to_state}</span></div>
          <div style="color:#94a3b8; margin-top:2px;">${escapeHtml(t.reason)}</div>
        </div>`
      )
      .join("");
  }
}

function updateWorkspaceBadge(folderPath) {
  if (!folderPath) return;
  const wsBar = document.getElementById("active-workspace-path-bar");
  if (wsBar) {
    wsBar.textContent = `📁 ${folderPath}`;
    wsBar.title = `Active Godot 4 Project Folder: ${folderPath}`;
  }
  const headerFolder = document.getElementById("header-folder-name");
  if (headerFolder) {
    const parts = String(folderPath).replace(/\\/g, "/").split("/").filter(Boolean);
    headerFolder.textContent = parts[parts.length - 1] || folderPath;
    headerFolder.title = folderPath;
  }
}

// ============================================================================
// Workspace Tree & Godot 4 3D Scene Graph Inspector
// ============================================================================
async function loadWorkspaceTree() {
  try {
    const res = await fetch("/api/workspace/tree");
    const data = await res.json();
    const container = document.getElementById("file-tree");
    if (!data.entries) return;

    const files = data.entries.filter((e) => e.type === "file");
    if (files.length > 0 && !files.some((f) => f.res_path === state.currentFileResPath)) {
      const preferred =
        files.find((f) => f.res_path === "res://scenes/main.tscn") ||
        files.find((f) => f.res_path === "res://scenes/main_3d.tscn") ||
        files.find((f) => f.res_path.endsWith(".tscn")) ||
        files.find((f) => f.res_path === "res://project.godot") ||
        files[0];
      if (preferred) {
        state.currentFileResPath = preferred.res_path;
        inspectWorkspaceFile(preferred.res_path);
      }
    }

    container.innerHTML = files
      .map((item) => {
        const icon = getFileIcon(item.name);
        const isActive = item.res_path === state.currentFileResPath ? "active" : "";
        return `
          <div class="tree-item ${isActive}" data-res-path="${escapeHtml(item.res_path)}">
            <span>${icon} ${escapeHtml(item.res_path.replace("res://", ""))}</span>
          </div>
        `;
      })
      .join("");

    container.querySelectorAll(".tree-item").forEach((el) => {
      el.addEventListener("click", () => {
        inspectWorkspaceFile(el.dataset.resPath);
        openRightDrawerTab("tab-diff");
      });
    });
  } catch (err) {
    console.error("Error loading workspace tree:", err);
  }
}

function getFileIcon(name) {
  const lower = name.toLowerCase();
  if (lower.endsWith(".tscn") || lower.endsWith(".tres")) return "🎬";
  if (lower.endsWith(".gd")) return "📜";
  if (lower.endsWith(".glb") || lower.endsWith(".blend")) return "🧊";
  if (lower.endsWith(".png") || lower.endsWith(".webp")) return "🖼️";
  if (lower.endsWith(".wav") || lower.endsWith(".ogg") || lower.endsWith(".mp3")) return "🔊";
  if (lower.endsWith(".godot")) return "⚙️";
  return "📄";
}

async function inspectWorkspaceFile(resPath) {
  state.currentFileResPath = resPath;
  document.querySelectorAll("#file-tree .tree-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.resPath === resPath);
  });

  try {
    const res = await fetch(`/api/workspace/file?path=${encodeURIComponent(resPath)}`);
    const data = await res.json();
    if (data.status === "success") {
      document.getElementById("file-viewer-code").textContent =
        data.numbered_content || "(empty file)";
      if (data.godot_scene_summary) {
        document.getElementById("active-scene-label").textContent = resPath.replace("res://", "");
        document.getElementById("scene-graph-summary").textContent = data.godot_scene_summary;
      }
    }
  } catch (err) {
    console.error("Failed to inspect file:", err);
  }
}

// ============================================================================
// Universal Drag-and-Drop, Drawers, Menu & UI Events
// ============================================================================
function openRightDrawerTab(tabId) {
  const rightDrawer = document.getElementById("right-drawer");
  if (rightDrawer) rightDrawer.classList.remove("hidden");
  document.querySelectorAll(".tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("hidden", p.id !== tabId);
  });
}

function toggleLeftDrawer(forceOpen = null) {
  const leftDrawer = document.getElementById("left-drawer");
  if (!leftDrawer) return;
  if (forceOpen === true) {
    leftDrawer.classList.remove("hidden");
  } else if (forceOpen === false) {
    leftDrawer.classList.add("hidden");
  } else {
    leftDrawer.classList.toggle("hidden");
  }
}

function bindUIEvents() {
  // Right-pane Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      openRightDrawerTab(btn.dataset.tab);
    });
  });

  // Left & Right Drawer Toggle Buttons
  const btnToggleFiles = document.getElementById("btn-toggle-files");
  if (btnToggleFiles) {
    btnToggleFiles.addEventListener("click", () => toggleLeftDrawer());
  }
  const btnCloseLeft = document.getElementById("btn-close-left-drawer");
  if (btnCloseLeft) {
    btnCloseLeft.addEventListener("click", () => toggleLeftDrawer(false));
  }
  const btnCloseRight = document.getElementById("btn-close-right-drawer");
  if (btnCloseRight) {
    btnCloseRight.addEventListener("click", () => {
      document.getElementById("right-drawer")?.classList.add("hidden");
    });
  }

  // Clickable Top Status Pill -> Opens VRAM & Engine Tab
  const btnStatusPill = document.getElementById("btn-status-pill");
  if (btnStatusPill) {
    btnStatusPill.addEventListener("click", () => openRightDrawerTab("tab-vram"));
  }

  // Top "☰ Menu & Tools" Dropdown
  const btnMenuToggle = document.getElementById("btn-menu-toggle");
  const dropdownMenu = document.getElementById("harness-dropdown-menu");
  if (btnMenuToggle && dropdownMenu) {
    btnMenuToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      dropdownMenu.classList.toggle("hidden");
    });
    window.addEventListener("click", () => {
      dropdownMenu.classList.add("hidden");
    });
    dropdownMenu.querySelectorAll(".dropdown-item").forEach((item) => {
      item.addEventListener("click", () => {
        dropdownMenu.classList.add("hidden");
        const act = item.dataset.action;
        if (act === "open-files") toggleLeftDrawer(true);
        else if (act === "open-folder-picker") openFolderModal();
        else if (act === "open-tab-diff") openRightDrawerTab("tab-diff");
        else if (act === "open-tab-studio") openRightDrawerTab("tab-studio");
        else if (act === "open-tab-vram") openRightDrawerTab("tab-vram");
        else if (act === "toggle-quick-chips") {
          document.getElementById("quick-actions-bar")?.classList.toggle("hidden");
        } else if (act === "open-setup") {
          openSetupModal();
        }
      });
    });
  }

  // Quick-action chips
  document.querySelectorAll(".chip-btn[data-prompt]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const promptInput = document.getElementById("prompt-input");
      promptInput.value = btn.dataset.prompt;
      sendChatPrompt(btn.dataset.prompt);
    });
  });

  // Test Tier 2 96k RAM Overflow button
  document.getElementById("btn-test-tier2").addEventListener("click", () => {
    sendChatPrompt(
      "Run a large-context architectural review of main_3d.tscn and player_controller.gd (Tier 2 96,000-token RAM Spillover test)",
      32000
    );
  });

  // Chat submit
  document.getElementById("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = document.getElementById("prompt-input");
    const text = input.value.trim();
    if (!text && state.attachments.length === 0) return;
    sendChatPrompt(text || "Inspect the attached files.");
  });

  // Enter to send (Shift+Enter for newline)
  document.getElementById("prompt-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      document.getElementById("chat-form").requestSubmit();
    }
  });

  // File input upload
  document.getElementById("file-input").addEventListener("change", async (e) => {
    if (e.target.files?.length) {
      await uploadAndIngestFiles(Array.from(e.target.files));
      e.target.value = "";
    }
  });

  // Universal Drag-and-Drop across workspace
  const dropZone = document.getElementById("drop-zone");
  const overlay = document.getElementById("drop-overlay");
  let dragDepth = 0;

  dropZone.addEventListener("dragenter", (e) => {
    e.preventDefault();
    dragDepth++;
    overlay.classList.remove("hidden");
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
  });

  dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) overlay.classList.add("hidden");
  });

  dropZone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dragDepth = 0;
    overlay.classList.add("hidden");
    if (e.dataTransfer?.files?.length) {
      await uploadAndIngestFiles(Array.from(e.dataTransfer.files));
    }
  });

  // Ctrl+V Clipboard Paste (Images or Text/Files)
  window.addEventListener("paste", async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    const filesToUpload = [];
    for (const item of items) {
      if (item.kind === "file") {
        const f = item.getAsFile();
        if (f) filesToUpload.push(f);
      }
    }
    if (filesToUpload.length > 0) {
      e.preventDefault();
      await uploadAndIngestFiles(filesToUpload);
    }
  });

  // Attach current inspected file to prompt
  document.getElementById("btn-attach-current-file").addEventListener("click", async () => {
    const res = await fetch("/api/ingest/path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.currentFileResPath }),
    });
    const data = await res.json();
    if (data.item) {
      state.attachments.push(data.item);
      renderAttachmentTray();
    }
  });

  // Model Profile Switcher
  document.getElementById("profile-select").addEventListener("change", async (e) => {
    await fetch("/api/vram/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile_id: e.target.value }),
    });
    await fetchStatus();
  });

  // Session Switcher (Tier 3 NVMe Disk Persistence ./cache/kv_disk/)
  const sessionSel = document.getElementById("session-select");
  if (sessionSel) {
    sessionSel.addEventListener("change", async (e) => {
      await fetch("/api/vram/session/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: e.target.value }),
      });
      await fetchStatus();
    });
  }

  // Approval Gate Toggle
  document.getElementById("approval-toggle").addEventListener("change", async (e) => {
    await fetch("/api/settings/approval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ require_approval: e.target.checked }),
    });
  });

  // Tier 5: Godot Play-Mode VRAM Yield button
  document.getElementById("btn-godot-yield").addEventListener("click", async () => {
    state.yieldedToGodot = !state.yieldedToGodot;
    await fetch(`/api/vram/yield?yield_to_godot=${state.yieldedToGodot}`, {
      method: "POST",
    });
    await fetchStatus();
  });

  // Refresh buttons
  document.getElementById("btn-refresh-tree").addEventListener("click", loadWorkspaceTree);
  document.getElementById("btn-refresh-diffs").addEventListener("click", loadDiffHistory);

  // Studio Direct 2D Generator
  document.getElementById("btn-studio-2d").addEventListener("click", async () => {
    const btn = document.getElementById("btn-studio-2d");
    btn.disabled = true;
    btn.textContent = "Swapping VRAM & Generating...";
    try {
      const res = await fetch("/api/tools/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tool_name: "generate_2d_asset",
          arguments: {
            prompt: document.getElementById("studio-2d-prompt").value,
            asset_type: document.getElementById("studio-2d-type").value,
            output_path: document.getElementById("studio-2d-path").value,
            transparent_bg: document.getElementById("studio-2d-rembg").checked,
            generate_pbr_maps: document.getElementById("studio-2d-pbr").checked,
          },
        }),
      });
      const data = await res.json();
      appendGalleryItem("2D Asset", data.result);
      await loadWorkspaceTree();
    } finally {
      btn.disabled = false;
      btn.textContent = "Generate 2D";
    }
  });

  // Studio Direct 3D Blender Generator
  document.getElementById("btn-studio-3d").addEventListener("click", async () => {
    const btn = document.getElementById("btn-studio-3d");
    btn.disabled = true;
    btn.textContent = "Running Headless Blender...";
    try {
      const res = await fetch("/api/tools/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tool_name: "run_blender_script",
          arguments: {
            script_code: document.getElementById("studio-3d-script").value,
            output_glb_path: document.getElementById("studio-3d-path").value,
            render_preview: true,
            heavy_cycles_bake: document.getElementById("studio-3d-cycles").checked,
          },
        }),
      });
      const data = await res.json();
      appendGalleryItem("3D Blender Asset", data.result);
      await loadWorkspaceTree();
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Blender Harness";
    }
  });

  // Studio Direct 2-in-1 Audio SFX/Music Generator
  const audioBtn = document.getElementById("btn-studio-audio");
  if (audioBtn) {
    audioBtn.addEventListener("click", async () => {
      audioBtn.disabled = true;
      audioBtn.textContent = "Swapping VRAM & Synthesizing...";
      try {
        const modeVal = document.getElementById("studio-audio-mode").value;
        const res = await fetch("/api/tools/execute", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tool_name: "generate_game_audio",
            arguments: {
              prompt: document.getElementById("studio-audio-prompt").value,
              mode: modeVal,
              output_path: document.getElementById("studio-audio-path").value,
              duration_sec: modeVal === "music_loop" ? 12.0 : 2.5,
              seamless_loop: document.getElementById("studio-audio-loop").checked,
            },
          }),
        });
        const data = await res.json();
        appendGalleryItem(`Audio (${modeVal})`, data.result);
        await loadWorkspaceTree();
      } finally {
        audioBtn.disabled = false;
        audioBtn.textContent = "Generate Audio";
      }
    });
  }

  // Project Folder Picker Modal
  const btnFolderModal = document.getElementById("btn-folder-modal");
  if (btnFolderModal) btnFolderModal.addEventListener("click", () => openFolderModal());
  const btnFolderInline = document.getElementById("btn-change-folder-inline");
  if (btnFolderInline) btnFolderInline.addEventListener("click", () => openFolderModal());
  const btnCloseFolder = document.getElementById("btn-close-folder-modal");
  if (btnCloseFolder) {
    btnCloseFolder.addEventListener("click", () => {
      document.getElementById("folder-modal").classList.add("hidden");
    });
  }

  // Setup & Health Check Modal
  document.getElementById("btn-setup-modal").addEventListener("click", openSetupModal);
  document.getElementById("btn-close-modal").addEventListener("click", () => {
    document.getElementById("setup-modal").classList.add("hidden");
  });
}

// ============================================================================
// Project Folder Picker, Creator & Directory Browser Modal
// ============================================================================
window.openFolderModal = async function (browsePath = null) {
  const modal = document.getElementById("folder-modal");
  const body = document.getElementById("folder-modal-body");
  modal.classList.remove("hidden");
  body.innerHTML = `<div style="padding:16px; color:#94a3b8;">Loading directories...</div>`;

  try {
    const [infoRes, browseRes] = await Promise.all([
      fetch("/api/workspace/info"),
      fetch("/api/workspace/browse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: browsePath }),
      }),
    ]);

    if (!infoRes.ok || !browseRes.ok) {
      body.innerHTML = `
        <div style="padding:14px; background:rgba(248,113,113,0.12); border:1px solid rgba(248,113,113,0.4); border-radius:8px; color:#fecaca; font-size:12.5px; line-height:1.55;">
          <strong>⚠️ Server Restart Required (HTTP ${browseRes.status})</strong><br/>
          Your open command prompt window is still running an older <code>server.py</code> process from before the Folder Picker endpoints were added.<br/><br/>
          <strong>Fix:</strong> Double-click <code>start_harness.bat</code> in <code>D:\\CodingProjects\\CustomHarness</code> (it will automatically stop the old server process on port 8000 and launch the updated server), then refresh this page.
        </div>
      `;
      return;
    }

    const info = await infoRes.json();
    const browse = await browseRes.json();
    if (info.workspace_dir) updateWorkspaceBadge(info.workspace_dir);

    const currentBrowsed = browse.current_dir || info.workspace_dir || "";
    const parentBrowsed = browse.parent_dir || currentBrowsed;

    const quickBtnsHtml = (browse.quick_paths || [])
      .map(
        (qp) =>
          `<button class="mini-btn js-browse-btn" data-path="${escapeHtml(qp.path)}">${escapeHtml(qp.label)}</button>`
      )
      .join(" ");

    const driveBtnsHtml = (browse.drives || [])
      .map(
        (d) =>
          `<button class="mini-btn js-browse-btn" data-path="${escapeHtml(d)}">💾 ${escapeHtml(d)}</button>`
      )
      .join(" ");

    const subdirRowsHtml = (browse.subdirs || [])
      .map((sd) => {
        const isCurrentActive =
          info.workspace_dir &&
          sd.path.toLowerCase() === String(info.workspace_dir).toLowerCase();
        return `
        <div class="setup-row" style="padding:8px 10px; ${isCurrentActive ? "border-color:#38bdf8; background:rgba(56,189,248,0.08);" : ""}">
          <div class="js-browse-btn" data-path="${escapeHtml(sd.path)}" style="cursor:pointer; flex:1;" title="Click to open inside this folder">
            <strong>📁 ${escapeHtml(sd.name)}</strong>
            ${sd.has_project_godot ? `<span style="color:#34d399; font-size:10.5px; margin-left:6px;">[Godot 4 Project]</span>` : ""}
            ${isCurrentActive ? `<span style="color:#38bdf8; font-size:10.5px; margin-left:6px;">[Active]</span>` : ""}
            <div style="color:#94a3b8; font-size:10.5px; font-family:var(--font-mono);">${escapeHtml(sd.path)}</div>
          </div>
          <div style="display:flex; gap:6px; align-items:center;">
            <button class="mini-btn js-browse-btn" data-path="${escapeHtml(sd.path)}">Open &rarr;</button>
            <button class="primary-btn js-select-btn" data-path="${escapeHtml(sd.path)}" style="padding:5px 10px; font-size:11.5px;">✓ Select Folder</button>
          </div>
        </div>`;
      })
      .join("");

    body.innerHTML = `
      <div id="folder-modal-alert" class="hidden" style="margin-bottom:10px; padding:8px 12px; border-radius:6px; font-size:12px;"></div>

      <!-- 1. Active Folder & Native Windows Picker -->
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; padding:10px 12px; background:#0f172a; border:1px solid #1e293b; border-radius:8px; margin-bottom:10px; flex-wrap:wrap;">
        <div style="min-width:220px; flex:1;">
          <div style="font-size:11px; color:#94a3b8;">Active Godot 4 Project Folder:</div>
          <div style="color:#38bdf8; font-family:var(--font-mono); font-size:12px; font-weight:600; word-break:break-all;">
            ${escapeHtml(info.workspace_dir)}
          </div>
        </div>
        <button id="btn-native-win-picker" class="primary-btn" style="padding:8px 14px; white-space:nowrap;">
          🪟 Browse / Create in Windows Explorer...
        </button>
      </div>

      <!-- 2. Create New Project Folder Inside Current Location -->
      <div style="padding:10px 12px; background:#0f172a; border:1px solid #1e293b; border-radius:8px; margin-bottom:10px;">
        <div style="font-size:11.5px; color:#e2e8f0; font-weight:600; margin-bottom:6px;">
          ➕ Create New Project Folder in <code style="color:#93c5fd;">${escapeHtml(currentBrowsed)}</code>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <input id="new-folder-name-input" type="text" placeholder="New folder name (e.g. NebulaBreaker)..." style="flex:1; padding:6px 10px; background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:6px; font-size:12px;" />
          <button id="btn-create-and-select-folder" class="primary-btn" style="padding:6px 12px; white-space:nowrap;">
            ➕ Create &amp; Use Folder
          </button>
          <button id="btn-create-only-folder" class="mini-btn" style="padding:6px 10px; white-space:nowrap;">
            + Create Only
          </button>
        </div>
      </div>

      <!-- 3. Navigation Bar: Up, Quick Jumps, Drives & Select Current Folder -->
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; flex-wrap:wrap;">
        <div style="display:flex; gap:5px; align-items:center; flex-wrap:wrap;">
          <button class="mini-btn js-browse-btn" data-path="${escapeHtml(parentBrowsed)}">⬆ Up One Level</button>
          ${quickBtnsHtml}
          ${driveBtnsHtml}
        </div>
        <button class="primary-btn js-select-btn" data-path="${escapeHtml(currentBrowsed)}" style="padding:5px 12px; font-size:11.5px; background:#059669;">
          ✓ Use Current Folder (${escapeHtml(currentBrowsed.split(/[\\/]/).filter(Boolean).pop() || currentBrowsed)})
        </button>
      </div>

      <!-- 4. Current Browsed Path Bar -->
      <div style="display:flex; gap:6px; align-items:center; margin-bottom:8px;">
        <span style="font-size:11px; color:#94a3b8; white-space:nowrap;">Browsing:</span>
        <input id="folder-manual-input" type="text" value="${escapeHtml(currentBrowsed)}" style="flex:1; padding:5px 8px; background:#1e293b; color:#cbd5e1; border:1px solid #334155; border-radius:5px; font-family:var(--font-mono); font-size:11.5px;" />
        <button id="btn-go-manual-path" class="mini-btn" style="padding:5px 10px;">Go</button>
        <button id="btn-select-manual-path" class="mini-btn" style="padding:5px 10px; color:#34d399; border-color:#059669;">✓ Set Active</button>
      </div>

      <!-- 5. Subdirectory List -->
      <div style="max-height:280px; overflow-y:auto; border:1px solid #1e293b; border-radius:8px; padding:6px; background:#090d16;">
        ${
          subdirRowsHtml ||
          `<div style="padding:16px; color:#94a3b8; font-size:12px; text-align:center;">
            No subfolders inside <code>${escapeHtml(currentBrowsed)}</code>.<br/>
            Click <strong>"✓ Use Current Folder"</strong> above or create a new folder using <strong>"➕ Create &amp; Use Folder"</strong>.
          </div>`
        }
      </div>
    `;

    // Bind all Browse buttons cleanly without inline string-escaping issues
    body.querySelectorAll(".js-browse-btn").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const p = el.getAttribute("data-path");
        if (p) openFolderModal(p);
      });
    });

    // Bind all Select buttons cleanly
    body.querySelectorAll(".js-select-btn").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.stopPropagation();
        const p = el.getAttribute("data-path");
        if (p) selectProjectFolder(p);
      });
    });

    // Bind Native Windows Explorer Folder Picker button
    const nativeBtn = document.getElementById("btn-native-win-picker");
    if (nativeBtn) {
      nativeBtn.addEventListener("click", () => pickNativeWindowsFolder(nativeBtn));
    }

    // Bind Create New Folder buttons
    const newFolderInput = document.getElementById("new-folder-name-input");
    const btnCreateSelect = document.getElementById("btn-create-and-select-folder");
    const btnCreateOnly = document.getElementById("btn-create-only-folder");

    if (btnCreateSelect && newFolderInput) {
      btnCreateSelect.addEventListener("click", () => {
        createNewProjectFolder(currentBrowsed, newFolderInput.value, true);
      });
      newFolderInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          createNewProjectFolder(currentBrowsed, newFolderInput.value, true);
        }
      });
    }
    if (btnCreateOnly && newFolderInput) {
      btnCreateOnly.addEventListener("click", () => {
        createNewProjectFolder(currentBrowsed, newFolderInput.value, false);
      });
    }

    // Bind manual path Go / Set Active
    const manualInput = document.getElementById("folder-manual-input");
    document.getElementById("btn-go-manual-path")?.addEventListener("click", () => {
      if (manualInput?.value) openFolderModal(manualInput.value.trim());
    });
    document.getElementById("btn-select-manual-path")?.addEventListener("click", () => {
      if (manualInput?.value) selectProjectFolder(manualInput.value.trim());
    });
  } catch (err) {
    body.innerHTML = `<div style="color:#f87171; padding:12px;">Failed to browse folders: ${escapeHtml(String(err))}</div>`;
  }
};

function showFolderModalAlert(msg, isError = true) {
  const el = document.getElementById("folder-modal-alert");
  if (!el) return;
  el.classList.remove("hidden");
  el.style.background = isError ? "rgba(248,113,113,0.14)" : "rgba(52,211,153,0.14)";
  el.style.border = isError ? "1px solid rgba(248,113,113,0.4)" : "1px solid rgba(52,211,153,0.4)";
  el.style.color = isError ? "#fecaca" : "#a7f3d0";
  el.textContent = msg;
}

window.createNewProjectFolder = async function (parentPath, newFolderName, selectImmediately = true) {
  const trimmed = (newFolderName || "").trim();
  if (!trimmed) {
    showFolderModalAlert("Please enter a name for the new folder first.", true);
    document.getElementById("new-folder-name-input")?.focus();
    return;
  }
  try {
    const res = await fetch("/api/workspace/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        parent_path: parentPath,
        new_folder_name: trimmed,
        select_immediately: selectImmediately,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      showFolderModalAlert(data.detail || `Failed to create folder (HTTP ${res.status})`, true);
      return;
    }
    if (selectImmediately && data.workspace_dir) {
      updateWorkspaceBadge(data.workspace_dir);
      if (data.telemetry) updateTelemetryUI(data.telemetry);
      document.getElementById("folder-modal").classList.add("hidden");
      await Promise.all([loadWorkspaceTree(), loadDiffHistory()]);
    } else {
      await openFolderModal(data.created_dir || parentPath);
    }
  } catch (err) {
    showFolderModalAlert(`Error creating folder: ${err}`, true);
  }
};

window.selectProjectFolder = async function (folderPath) {
  if (!folderPath || !folderPath.trim()) return;
  try {
    const res = await fetch("/api/workspace/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_path: folderPath.trim(), init_starter_if_empty: false }),
    });
    const data = await res.json();
    if (!res.ok) {
      showFolderModalAlert(data.detail || `Failed to switch folder (HTTP ${res.status})`, true);
      return;
    }
    if (data.workspace_dir) updateWorkspaceBadge(data.workspace_dir);
    if (data.telemetry) updateTelemetryUI(data.telemetry);
    document.getElementById("folder-modal").classList.add("hidden");
    await Promise.all([loadWorkspaceTree(), loadDiffHistory()]);
  } catch (err) {
    showFolderModalAlert(`Failed to switch folder: ${err}`, true);
  }
};

window.pickNativeWindowsFolder = async function (btnEl = null) {
  const origText = btnEl ? btnEl.textContent : "";
  if (btnEl) {
    btnEl.disabled = true;
    btnEl.textContent = "⏳ Waiting for Windows Folder Dialog...";
  }
  try {
    const res = await fetch("/api/workspace/pick_native", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      showFolderModalAlert(data.detail || `Windows Folder Dialog error (HTTP ${res.status})`, true);
      return;
    }
    if (data.status === "success" && data.workspace_dir) {
      updateWorkspaceBadge(data.workspace_dir);
      if (data.telemetry) updateTelemetryUI(data.telemetry);
      document.getElementById("folder-modal").classList.add("hidden");
      await Promise.all([loadWorkspaceTree(), loadDiffHistory()]);
    }
  } catch (err) {
    showFolderModalAlert(`Native folder dialog failed: ${err}`, true);
  } finally {
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.textContent = origText;
    }
  }
};

async function uploadAndIngestFiles(files) {
  const formData = new FormData();
  for (const f of files) {
    formData.append("files", f);
  }
  try {
    const res = await fetch("/api/ingest", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (Array.isArray(data.items)) {
      state.attachments.push(...data.items);
      renderAttachmentTray();
    }
  } catch (err) {
    console.error("Failed to ingest dropped/pasted files:", err);
  }
}

function renderAttachmentTray() {
  const tray = document.getElementById("attachment-tray");
  if (state.attachments.length === 0) {
    tray.classList.add("hidden");
    tray.innerHTML = "";
    return;
  }
  tray.classList.remove("hidden");
  tray.innerHTML = state.attachments
    .map(
      (att, idx) => `
      <div class="attachment-pill" title="${escapeHtml(att.summary_text || "")}">
        <span>📎 ${escapeHtml(att.filename)} (${escapeHtml(att.category || "file")})</span>
        <button type="button" data-idx="${idx}">✕</button>
      </div>`
    )
    .join("");

  tray.querySelectorAll("button[data-idx]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.attachments.splice(Number(btn.dataset.idx), 1);
      renderAttachmentTray();
    });
  });
}

// ============================================================================
// Antigravity / Claude Code / Codex Expandable Process Rows (>)
// ============================================================================
function describeToolAction(toolName, args = {}, targetHint = "") {
  const path =
    targetHint ||
    args.output_path ||
    args.output_glb_path ||
    args.path ||
    args.command ||
    args.query ||
    "";
  const map = {
    generate_2d_asset: "Generate 2D texture / sprite",
    run_blender_script: "Model 3D .glb asset in Blender",
    generate_game_audio: `Synthesize ${args.mode === "music_loop" ? "music loop" : "sound effect"}`,
    write_file: "Write file",
    apply_diff: "Edit file (SEARCH/REPLACE diff)",
    read_file: "Read file",
    list_dir: "List workspace directory",
    recall_memory: "Recall project memory (0-VRAM RAG)",
    run_command: "Run validation / command",
  };
  return {
    title: map[toolName] || `Run ${toolName}`,
    target: path,
  };
}

function handleLiveStepStart(data) {
  const statusHeader = document.getElementById("active-progress-header");
  const sendBtn = document.getElementById("btn-send");
  if (statusHeader && data) {
    statusHeader.textContent = `Step ${data.step} — Planning & executing tools...`;
  }
  if (sendBtn && sendBtn.disabled && data) {
    sendBtn.textContent = `Step ${data.step}...`;
  }
}

function handleLiveToolStart(data) {
  if (!data) return;
  const listEl = document.getElementById("active-progress-list");
  const statusHeader = document.getElementById("active-progress-header");
  const sendBtn = document.getElementById("btn-send");
  const desc = describeToolAction(data.name, data.arguments || {}, data.target_hint || "");

  if (statusHeader) {
    statusHeader.textContent = `Step ${data.step || 1} (${data.tool_index || 1}/${data.tool_count || 1}) · ${desc.title}`;
  }
  if (sendBtn && sendBtn.disabled) {
    sendBtn.textContent = `Running...`;
  }
  if (listEl) {
    const details = document.createElement("details");
    details.id = `live-tool-${data.id}`;
    details.className = "process-row";
    details.innerHTML = `
      <summary class="process-summary">
        <span class="process-left">
          <span class="process-icon process-icon-run" id="live-icon-${data.id}">⏳</span>
          <span class="process-title">${escapeHtml(desc.title)}</span>
          ${desc.target ? `<code class="process-target">${escapeHtml(desc.target)}</code>` : ""}
        </span>
        <span class="process-right">
          <span class="process-meta" id="live-meta-${data.id}">Running...</span>
          <span class="process-chevron">›</span>
        </span>
      </summary>
      <div class="process-details" id="live-details-${data.id}">Tool: ${escapeHtml(data.name)}\nArguments: ${escapeHtml(JSON.stringify(data.arguments || {}, null, 2))}</div>
    `;
    listEl.appendChild(details);
    const stream = document.getElementById("chat-stream");
    if (stream) stream.scrollTop = stream.scrollHeight;
  }
}

function handleLiveToolComplete(data) {
  if (!data) return;
  const iconEl = document.getElementById(`live-icon-${data.id}`);
  const metaEl = document.getElementById(`live-meta-${data.id}`);
  const detailsEl = document.getElementById(`live-details-${data.id}`);
  const ok = data.result?.status !== "error";
  if (iconEl) {
    iconEl.className = `process-icon ${ok ? "process-icon-ok" : "process-icon-err"}`;
    iconEl.textContent = ok ? "✓" : "✗";
  }
  if (metaEl) {
    metaEl.textContent = data.elapsed_sec != null ? `${data.elapsed_sec}s` : "Done";
  }
  if (detailsEl && data.result) {
    const previewHtml = buildToolPreviewHtml(data.name, data.result);
    if (previewHtml) {
      detailsEl.innerHTML += previewHtml;
    }
  }
  if (
    data.result &&
    (data.name === "generate_2d_asset" ||
      data.name === "run_blender_script" ||
      data.name === "generate_game_audio")
  ) {
    appendGalleryItem(data.name, data.result);
  }
}

async function sendChatPrompt(promptText, simulateContextTokens = null) {
  const stream = document.getElementById("chat-stream");
  const input = document.getElementById("prompt-input");
  const sendBtn = document.getElementById("btn-send");

  const currentAttachments = [...state.attachments];
  state.attachments = [];
  renderAttachmentTray();
  input.value = "";

  // Append User Message Card
  const userDiv = document.createElement("div");
  userDiv.className = "msg msg-user";
  const attBadges = currentAttachments
    .map((a) => `<code>📎 ${escapeHtml(a.filename)}</code>`)
    .join(" ");
  userDiv.innerHTML = `
    <div class="msg-header">
      <span class="msg-role">YOU</span>
      <span class="msg-meta">${new Date().toLocaleTimeString()}</span>
    </div>
    <div class="msg-body">${escapeHtml(promptText)}${attBadges ? `<div style="margin-top:6px">${attBadges}</div>` : ""}</div>
  `;
  stream.appendChild(userDiv);

  // Append Live Antigravity-Style Expandable Process Card
  const progressCard = document.createElement("div");
  progressCard.id = "active-progress-card";
  progressCard.className = "msg msg-assistant";
  progressCard.innerHTML = `
    <div class="msg-header">
      <span class="msg-role" style="color:#60a5fa;">● WORKING</span>
      <span class="msg-meta" id="active-progress-header">Initializing step 1...</span>
    </div>
    <div id="active-progress-list" class="process-list"></div>
  `;
  stream.appendChild(progressCard);
  progressCard.scrollIntoView({ behavior: "smooth", block: "end" });

  sendBtn.disabled = true;
  sendBtn.textContent = "Running...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptText,
        attachments: currentAttachments,
        simulate_context_tokens: simulateContextTokens,
      }),
    });
    const data = await res.json();
    const existingProg = document.getElementById("active-progress-card");
    if (existingProg) existingProg.remove();
    const assistantCard = appendAssistantTurnCard(data);
    if (assistantCard) {
      assistantCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (data.vram_telemetry) updateTelemetryUI(data.vram_telemetry);
    await Promise.all([loadWorkspaceTree(), loadDiffHistory()]);
  } catch (err) {
    console.error("Chat error:", err);
    const existingProg = document.getElementById("active-progress-card");
    if (existingProg) {
      existingProg.innerHTML = `<div style="color:#f87171;">✗ Error executing turn: ${escapeHtml(String(err))}</div>`;
    }
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send ↵";
  }
}

function appendAssistantTurnCard(data) {
  const stream = document.getElementById("chat-stream");
  const card = document.createElement("div");
  card.className = "msg msg-assistant";

  const thinkRowHtml = data.thinking
    ? `
      <details class="process-row">
        <summary class="process-summary">
          <span class="process-left">
            <span class="process-icon process-icon-think">💭</span>
            <span class="process-title">Reasoning trace</span>
            <code class="process-target">&lt;think&gt;...&lt;/think&gt;</code>
          </span>
          <span class="process-right">
            <span class="process-meta">Expand</span>
            <span class="process-chevron">›</span>
          </span>
        </summary>
        <div class="process-details">${escapeHtml(data.thinking)}</div>
      </details>`
    : "";

  const toolRowsHtml = (data.tool_calls || [])
    .map((tc) => renderToolCallCard(tc))
    .join("");

  const processListHtml =
    thinkRowHtml || toolRowsHtml
      ? `<div class="process-list">${thinkRowHtml}${toolRowsHtml}</div>`
      : "";

  const compactionBanner = data.tier2_compaction
    ? `<div class="badge-vision-check" style="margin-bottom:8px;">
         ✓ Tier 2 (96k RAM Spillover) Turn Completed &rarr; Compacted to ${data.tier2_compaction.compacted_tokens} tokens &amp; Restored Tier 1 Fast VRAM Mode
       </div>`
    : "";

  const toolCount = (data.tool_calls || []).length;
  card.innerHTML = `
    <div class="msg-header">
      <span class="msg-role">GODOT 4 HARNESS</span>
      <span class="msg-meta">${toolCount > 0 ? `${toolCount} process step${toolCount > 1 ? "s" : ""} · ` : ""}${new Date().toLocaleTimeString()}</span>
    </div>
    ${compactionBanner}
    ${processListHtml}
    <div class="msg-body">${formatMarkdownLite(data.content || "")}</div>
  `;

  stream.appendChild(card);
  return card;
}

function buildToolPreviewHtml(toolName, res = {}) {
  if (toolName === "generate_2d_asset" && res.closed_loop_vision_check?.image_url) {
    const pbr = res.pbr_maps || {};
    return `
      <div class="preview-card">
        <div>
          <span class="badge-vision-check">✓ Closed-Loop Vision Self-Check Passed</span>
          <code style="margin-left:6px;">${escapeHtml(res.res_path || "")}</code>
        </div>
        <img src="${res.closed_loop_vision_check.image_url}" alt="Generated 2D Asset Preview" />
        ${
          pbr.normal_map
            ? `<div style="margin-top:6px; font-size:11px; color:#94a3b8;">
                 PBR Maps: <code>${escapeHtml(pbr.normal_map)}</code> |
                 <code>${escapeHtml(pbr.roughness_map)}</code> |
                 <code>${escapeHtml(pbr.godot_material_tres)}</code>
               </div>`
            : ""
        }
      </div>
    `;
  }
  if (toolName === "run_blender_script" && res.contact_sheet?.image_url) {
    return `
      <div class="preview-card">
        <div>
          <span class="badge-vision-check">✓ 3-Angle Visual Verification Contact Sheet (Front -Y | 3/4 Iso | Wireframe/Top)</span>
          <code style="margin-left:6px;">${escapeHtml(res.res_glb_path || "")}</code>
        </div>
        <img src="${res.contact_sheet.image_url}" alt="3-Angle Blender Contact Sheet" />
      </div>
    `;
  }
  if (toolName === "generate_game_audio" && res.closed_loop_vision_check?.image_url) {
    return `
      <div class="preview-card">
        <div>
          <span class="badge-vision-check">✓ Waveform + Mel-Spectrogram Vision Check Passed</span>
          <code style="margin-left:6px;">${escapeHtml(res.res_path || "")}</code>
        </div>
        <img src="${res.closed_loop_vision_check.image_url}" alt="Audio Waveform and Mel-Spectrogram" />
      </div>
    `;
  }
  if (res.status === "approval_required" && res.cmd_id) {
    return `
      <div style="margin-top:8px;">
        <button class="mini-btn mini-btn-accent" onclick="approveCommand('${res.cmd_id}')">
          ✓ Approve &amp; Run Command: ${escapeHtml(res.command)}
        </button>
      </div>
    `;
  }
  return "";
}

function renderToolCallCard(tc) {
  const res = tc.result || {};
  const ok = res.status !== "error";
  const desc = describeToolAction(tc.name, tc.arguments || {}, tc.target_hint || "");
  const elapsed = tc.elapsed_sec != null ? `${tc.elapsed_sec}s` : res.status || "done";
  const previewHtml = buildToolPreviewHtml(tc.name, res);

  return `
    <details class="process-row">
      <summary class="process-summary">
        <span class="process-left">
          <span class="process-icon ${ok ? "process-icon-ok" : "process-icon-err"}">${ok ? "✓" : "✗"}</span>
          <span class="process-title">${escapeHtml(desc.title)}</span>
          ${desc.target ? `<code class="process-target">${escapeHtml(desc.target)}</code>` : ""}
        </span>
        <span class="process-right">
          <span class="process-meta">${escapeHtml(elapsed)}</span>
          <span class="process-chevron">›</span>
        </span>
      </summary>
      <div class="process-details">Tool: ${escapeHtml(tc.name)}\nArgs: ${escapeHtml(JSON.stringify(tc.arguments || {}, null, 2))}${previewHtml}</div>
    </details>
  `;
}

window.approveCommand = async function (cmdId) {
  await fetch("/api/commands/approve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cmd_id: cmdId }),
  });
  await loadWorkspaceTree();
};

function appendGalleryItem(label, result) {
  if (!result) return;
  const gallery = document.getElementById("asset-gallery");
  const imgUrl =
    result.closed_loop_vision_check?.image_url || result.contact_sheet?.image_url;
  if (!imgUrl) return;

  const div = document.createElement("div");
  div.className = "preview-card";
  const pathLabel = result.res_path || result.res_glb_path || label;
  div.innerHTML = `
    <div style="display:flex; justify-content:space-between; font-size:11px; font-family:var(--font-mono);">
      <span class="badge-vision-check">${escapeHtml(label)}</span>
      <code>${escapeHtml(pathLabel)}</code>
    </div>
    <img src="${imgUrl}" alt="${escapeHtml(pathLabel)}" />
  `;
  gallery.prepend(div);
}

// ============================================================================
// Live Code Diff Viewer & .bak Rollback
// ============================================================================
async function loadDiffHistory() {
  try {
    const res = await fetch("/api/diff/history");
    const data = await res.json();
    const list = document.getElementById("diff-list");
    const diffs = data.diffs || [];
    if (diffs.length === 0) {
      list.innerHTML = `<div class="empty-hint">No edits applied yet. Run <code>apply_diff</code> or <code>write_file</code> to inspect unified diffs and one-click <code>.bak</code> rollbacks.</div>`;
      return;
    }

    list.innerHTML = diffs
      .map((d) => {
        const coloredDiff = (d.unified_diff || "")
          .split("\n")
          .map((line) => {
            if (line.startsWith("+") && !line.startsWith("+++")) {
              return `<span class="diff-line-add">${escapeHtml(line)}</span>`;
            }
            if (line.startsWith("-") && !line.startsWith("---")) {
              return `<span class="diff-line-del">${escapeHtml(line)}</span>`;
            }
            return escapeHtml(line);
          })
          .join("\n");

        return `
          <div class="diff-card">
            <div class="diff-card-header">
              <span><strong>${escapeHtml(d.res_path)}</strong> (+${d.lines_added} / -${d.lines_removed})</span>
              ${
                d.rolled_back
                  ? `<span style="color:#f59e0b;">Rolled Back</span>`
                  : `<button class="mini-btn" onclick="rollbackDiff('${d.diff_id}')">↩ Rollback .bak</button>`
              }
            </div>
            <pre class="diff-pre">${coloredDiff || "(No line changes)"}</pre>
          </div>
        `;
      })
      .join("");
  } catch (err) {
    console.error("Failed to load diff history:", err);
  }
}

window.rollbackDiff = async function (diffId) {
  await fetch("/api/diff/rollback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ diff_id: diffId }),
  });
  await Promise.all([loadDiffHistory(), inspectWorkspaceFile(state.currentFileResPath)]);
};

// ============================================================================
// First-Run Setup & Health Check Modal + Live Downloader
// ============================================================================
function handleDownloadProgress(job) {
  if (!job || !job.job_key) return;
  const safeId = job.job_key.replace(/[^a-zA-Z0-9_-]/g, "_");
  const statusEl = document.getElementById(`dl-status-${safeId}`);
  if (statusEl) {
    if (job.status === "downloading") {
      const pct = job.percent != null ? `${job.percent}%` : "";
      const mb = job.downloaded_mb != null ? `${job.downloaded_mb} MB` : "";
      const tot = job.total_mb != null ? ` / ${job.total_mb} MB` : "";
      const spd = job.speed_mbps != null ? ` (${job.speed_mbps} MB/s)` : "";
      statusEl.innerHTML = `<span style="color:#38bdf8;">⬇ ${escapeHtml(job.current_file || "")} ${pct} — ${mb}${tot}${spd}</span>`;
    } else if (job.status === "completed") {
      statusEl.innerHTML = `<span style="color:#34d399;">✓ ${escapeHtml(job.message || "Download Complete")}</span>`;
      const modal = document.getElementById("setup-modal");
      if (modal && !modal.classList.contains("hidden")) {
        setTimeout(() => {
          const savedToken = document.getElementById("setup-hf-token")?.value || "";
          openSetupModal().then(() => {
            const inputAfter = document.getElementById("setup-hf-token");
            if (inputAfter && savedToken) inputAfter.value = savedToken;
          });
        }, 500);
      }
    } else if (job.status === "error") {
      statusEl.innerHTML = `<span style="color:#f87171;">✗ ${escapeHtml(job.error || "Download Failed")}</span>`;
    } else {
      statusEl.innerHTML = `<span style="color:#fbbf24;">⏳ ${escapeHtml(job.message || job.status)}</span>`;
    }
  }
}

window.triggerSetupDownload = async function (category, itemId) {
  const hfInput = document.getElementById("setup-hf-token");
  const hfToken = hfInput ? hfInput.value.trim() : "";
  try {
    await fetch("/api/setup/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        category,
        item_id: itemId,
        execute_download: true,
        hf_token: hfToken || null,
      }),
    });
  } catch (err) {
    console.error("Failed to start download:", err);
  }
};

async function openSetupModal() {
  const modal = document.getElementById("setup-modal");
  const body = document.getElementById("setup-modal-body");
  modal.classList.remove("hidden");
  body.innerHTML = "Running Windows 11 CUDA, GGUF & Diffusers health check...";

  try {
    const res = await fetch("/api/setup/check");
    const data = await res.json();

    const binRows = Object.entries(data.binaries || {})
      .map(([k, v]) => {
        const canDownload = k === "llama_server" || k === "stable_diffusion";
        const itemId = v.id || (k === "llama_server" ? "llama-server" : "sd");
        const jobId = `binary_${itemId}`.replace(/[^a-zA-Z0-9_-]/g, "_");
        return `
        <div class="setup-row">
          <div>
            <strong>${escapeHtml(k)}</strong>
            <div style="color:#94a3b8; font-size:10.5px;">${escapeHtml(v.path || "Not found on PATH")}</div>
            <div id="dl-status-${jobId}" style="font-size:10.5px; margin-top:2px;"></div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="color:${v.exists ? "#34d399" : "#fbbf24"}">
              ${v.exists ? "✓ Ready" : "⚡ Portable / Fallback Active"}
            </span>
            ${
              canDownload && !v.exists
                ? `<button class="mini-btn" onclick="triggerSetupDownload('binary', '${itemId}')">⬇ Download CUDA .zip</button>`
                : ""
            }
          </div>
        </div>`;
      })
      .join("");

    const llmRows = Object.values(data.llm_models || {})
      .map((m) => {
        const jobId = `llm_${m.id}`.replace(/[^a-zA-Z0-9_-]/g, "_");
        return `
        <div class="setup-row">
          <div>
            <strong>${escapeHtml(m.display_name)}</strong>
            <div style="color:#94a3b8; font-size:10.5px;">${escapeHtml(m.model_path)}</div>
            <div id="dl-status-${jobId}" style="font-size:10.5px; margin-top:2px;"></div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="color:${m.model_exists ? "#34d399" : "#93c5fd"}">
              ${m.model_exists ? `✓ ${m.model_size_mb} MB` : "GGUF Target Configured"}
            </span>
            ${
              !m.model_exists
                ? `<button class="mini-btn" onclick="triggerSetupDownload('llm', '${m.id}')">⬇ Download GGUF + mmproj</button>`
                : ""
            }
          </div>
        </div>`;
      })
      .join("");

    const imgRows = Object.values(data.image_models || {})
      .map((m) => {
        const jobId = `image_model_${m.id}`.replace(/[^a-zA-Z0-9_-]/g, "_");
        return `
        <div class="setup-row">
          <div>
            <strong>${escapeHtml(m.display_name)}</strong>
            <div style="color:#94a3b8; font-size:10.5px;">${escapeHtml(m.path)}</div>
            <div id="dl-status-${jobId}" style="font-size:10.5px; margin-top:2px;"></div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="color:${m.exists ? "#34d399" : "#93c5fd"}">
              ${m.exists ? `✓ ${m.size_mb} MB` : "4-Step 1024x1024 Configured"}
            </span>
            ${
              !m.exists
                ? `<button class="mini-btn" onclick="triggerSetupDownload('image_model', '${m.id}')">⬇ Download GGUF</button>`
                : ""
            }
          </div>
        </div>`;
      })
      .join("");

    const audioRows = Object.values(data.audio_models || {})
      .map((m) => {
        const jobId = `audio_model_${m.id}`.replace(/[^a-zA-Z0-9_-]/g, "_");
        return `
        <div class="setup-row">
          <div>
            <strong>${escapeHtml(m.display_name)}</strong>
            <div style="color:#94a3b8; font-size:10.5px;">${escapeHtml(m.path)} (${escapeHtml(m.repo_id)})</div>
            ${!m.exists && m.gated_note ? `<div style="color:#eab308; font-size:10px;">${escapeHtml(m.gated_note)}</div>` : ""}
            <div id="dl-status-${jobId}" style="font-size:10.5px; margin-top:2px;"></div>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span style="color:${m.exists ? "#34d399" : "#93c5fd"}">
              ${m.exists ? "✓ Ready (FP16)" : "Diffusers FP16 Target"}
            </span>
            ${
              !m.exists
                ? `<button class="mini-btn" onclick="triggerSetupDownload('audio_model', '${m.id}')">⬇ Download Weights</button>`
                : ""
            }
          </div>
        </div>`;
      })
      .join("");

    body.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px; padding:8px; background:#0f172a; border:1px solid #1e293b; border-radius:6px;">
        <div style="flex:1;">
          <label style="font-size:11px; color:#94a3b8; display:block; margin-bottom:3px;">
            Optional Hugging Face Token (<code>HF_TOKEN</code> — required for gated repos like <code>stabilityai/stable-audio-open-1.0</code>):
          </label>
          <input id="setup-hf-token" type="password" placeholder="hf_..." style="width:100%; padding:4px 8px; background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:4px; font-size:11.5px;" />
        </div>
        <button class="primary-btn" style="white-space:nowrap; padding:7px 12px;" onclick="triggerSetupDownload('all_defaults', 'all')">
          ⬇ Download All Default Models &amp; Binaries
        </button>
      </div>
      <h4>1. Local GPU &amp; Portable Binaries (./bin)</h4>
      ${binRows}
      <h4>2. Local VLM GGUF Profiles (./models)</h4>
      ${llmRows}
      <h4>3. Local 2D Image-Gen GGUF Weights (./models)</h4>
      ${imgRows}
      <h4>4. Local 2-in-1 Audio-Gen Weights (./models/stable-audio-open-1.0)</h4>
      ${audioRows}
    `;
  } catch (err) {
    body.innerHTML = `<div style="color:#f87171">Health check error: ${escapeHtml(String(err))}</div>`;
  }
}

// ============================================================================
// Helpers
// ============================================================================
function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatMarkdownLite(text) {
  let html = escapeHtml(text);
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre class="code-viewer" style="margin-top:8px; border-radius:5px;">${code}</pre>`;
  });
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return html;
}
