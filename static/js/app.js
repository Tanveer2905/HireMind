/**
 * app.js — AI Hiring Copilot Frontend Logic
 * Handles resume upload, skill management, analysis pipeline, result display,
 * LLM reranking, chat interface, and feedback system.
 */

// ============================================================
// Auth Interceptor & State
// ============================================================
const originalFetch = window.fetch;
window.fetch = async function() {
    let [resource, config] = arguments;
    if (config === undefined) config = {};
    const token = localStorage.getItem("access_token");
    if (token) {
        config.headers = config.headers || {};
        config.headers["Authorization"] = `Bearer ${token}`;
    }
    const res = await originalFetch(resource, config);
    if (res.status === 401) {
        localStorage.removeItem("access_token");
        if (typeof checkAuth === "function") {
            checkAuth();
        }
    }
    return res;
};

// ============================================================
// State
// ============================================================
const state = {
    mustHaveSkills: [],
    results: null,
    analysisData: null,
    llmAvailable: false,
    chatOpen: false,
};

// ============================================================
// DOM References
// ============================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    jdInput:          $("#jdInput"),
    skillInput:       $("#skillInput"),
    addSkillBtn:      $("#addSkillBtn"),
    skillChips:       $("#skillChips"),
    dropzone:         $("#dropzone"),
    fileInput:        $("#fileInput"),
    resumeList:       $("#resumeList"),
    resumeCountText:  $("#resumeCountText"),
    analyzeBtn:       $("#analyzeBtn"),
    loadingSection:   $("#loadingSection"),
    loadingText:      $("#loadingText"),
    resultsSection:   $("#resultsSection"),
    statsBar:         $("#statsBar"),
    candidatesGrid:   $("#candidatesGrid"),
    detectedSkills:   $("#detectedSkillsArea"),
    modalOverlay:     $("#modalOverlay"),
    modalContent:     $("#modalContent"),
    modalClose:       $("#modalClose"),
    llmToggle:        $("#llmToggle"),
    llmToggleLabel:   $("#llmToggleLabel"),
    llmDot:           $("#llmDot"),
    llmStatusText:    $("#llmStatusText"),
    chatToggleBtn:    $("#chatToggleBtn"),
    chatDrawer:       $("#chatDrawer"),
    chatCloseBtn:     $("#chatCloseBtn"),
    chatMessages:     $("#chatMessages"),
    chatInput:        $("#chatInput"),
    chatSendBtn:      $("#chatSendBtn"),
    statLLMContainer: $("#statLLMContainer"),
    authModal:        $("#authModal"),
    authForm:         $("#authForm"),
    authEmail:        $("#authEmail"),
    authPassword:     $("#authPassword"),
    authSubmitBtn:    $("#authSubmitBtn"),
    authError:        $("#authError"),
    tabLogin:         $("#tabLogin"),
    tabRegister:      $("#tabRegister"),
    logoutBtn:        $("#logoutBtn"),
    downloadExcelBtn: $("#downloadExcelBtn"),
};

// ============================================================
// Init
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    checkAuth();
    loadResumes();
    checkLLMStatus();
    setupEventListeners();
    setupDropzone();
    setup3DCards();
});

// ============================================================
// Auth Flow
// ============================================================
let isLoginMode = true;

function checkAuth() {
    const token = localStorage.getItem("access_token");
    if (!token) {
        dom.authModal.classList.add("modal-overlay--active");
        dom.logoutBtn.style.display = "none";
    } else {
        dom.authModal.classList.remove("modal-overlay--active");
        dom.logoutBtn.style.display = "inline-flex";
    }
}

dom.tabLogin?.addEventListener("click", () => {
    isLoginMode = true;
    dom.tabLogin.style.borderBottomColor = "var(--accent-purple)";
    dom.tabLogin.style.color = "white";
    dom.tabRegister.style.borderBottomColor = "transparent";
    dom.tabRegister.style.color = "rgba(255,255,255,0.5)";
    dom.authSubmitBtn.textContent = "Login";
    dom.authError.style.display = "none";
});

dom.tabRegister?.addEventListener("click", () => {
    isLoginMode = false;
    dom.tabRegister.style.borderBottomColor = "var(--accent-purple)";
    dom.tabRegister.style.color = "white";
    dom.tabLogin.style.borderBottomColor = "transparent";
    dom.tabLogin.style.color = "rgba(255,255,255,0.5)";
    dom.authSubmitBtn.textContent = "Register";
    dom.authError.style.display = "none";
});

dom.authForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    dom.authSubmitBtn.disabled = true;
    dom.authError.style.display = "none";
    
    const email = dom.authEmail.value.trim();
    const password = dom.authPassword.value.trim();
    const endpoint = isLoginMode ? "/api/login" : "/api/register";
    
    try {
        const res = await originalFetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password })
        });
        const data = await res.json();
        
        if (!res.ok) {
            throw new Error(data.detail || "Authentication failed");
        }
        
        if (isLoginMode) {
            localStorage.setItem("access_token", data.access_token);
            window.location.reload();
        } else {
            showToast("Registration successful! Please login.", "success");
            dom.tabLogin.click();
            dom.authPassword.value = "";
        }
    } catch (err) {
        dom.authError.textContent = err.message;
        dom.authError.style.display = "block";
    } finally {
        dom.authSubmitBtn.disabled = false;
    }
});

dom.logoutBtn?.addEventListener("click", () => {
    localStorage.removeItem("access_token");
    window.location.reload();
});

// ============================================================
// Event Listeners
// ============================================================
function setupEventListeners() {
    // Add skill
    dom.addSkillBtn.addEventListener("click", addSkill);
    dom.skillInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") addSkill();
    });

    // Analyze
    dom.analyzeBtn.addEventListener("click", runAnalysis);

    // Download Excel
    dom.downloadExcelBtn?.addEventListener("click", downloadExcel);

    // Modal
    dom.modalClose.addEventListener("click", closeModal);
    dom.modalOverlay.addEventListener("click", (e) => {
        if (e.target === dom.modalOverlay) closeModal();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeModal();
            if (state.chatOpen) toggleChat();
        }
    });

    // Chat
    dom.chatToggleBtn.addEventListener("click", toggleChat);
    dom.chatCloseBtn.addEventListener("click", toggleChat);
    dom.chatSendBtn.addEventListener("click", sendChatMessage);
    dom.chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendChatMessage();
    });

    // Nav scroll effect
    window.addEventListener("scroll", () => {
        const nav = $("#nav");
        nav.style.background = window.scrollY > 20
            ? "rgba(6, 6, 15, 0.85)"
            : "rgba(6, 6, 15, 0.6)";
    });
}

// ============================================================
// LLM Status
// ============================================================
async function checkLLMStatus() {
    try {
        const res = await fetch("/api/llm/status");
        const data = await res.json();

        state.llmAvailable = data.available;

        if (data.available) {
            dom.llmDot.className = "nav__badge-dot";
            dom.llmStatusText.textContent = "LLM Ready";
            dom.llmToggle.disabled = false;
            dom.llmToggleLabel.title = "Enable LLM-powered AI reasoning";
        } else {
            dom.llmDot.className = "nav__badge-dot nav__badge-dot--grey";
            dom.llmStatusText.textContent = "LLM Offline";
            dom.llmToggle.disabled = true;
            dom.llmToggle.checked = false;
            dom.llmToggleLabel.title = "Ollama not running — install from ollama.com";
        }
    } catch (err) {
        dom.llmDot.className = "nav__badge-dot nav__badge-dot--grey";
        dom.llmStatusText.textContent = "LLM Offline";
        dom.llmToggle.disabled = true;
    }
}

// ============================================================
// 3D Card Tilt Effect
// ============================================================
function setup3DCards() {
    const apply3DEffect = (cards) => {
        cards.forEach((card) => {
            // Prevent multiple listeners
            if (card.dataset.has3d) return;
            card.dataset.has3d = "true";

            card.addEventListener("mousemove", (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = ((y - centerY) / centerY) * -5;
                const rotateY = ((x - centerX) / centerX) * 5;

                card.style.transform =
                    `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
                
                // Add a dynamic glare effect
                card.style.backgroundImage = `radial-gradient(circle at ${x}px ${y}px, rgba(255,255,255,0.08) 0%, transparent 60%)`;
            });

            card.addEventListener("mouseleave", () => {
                card.style.transform = "perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)";
                card.style.backgroundImage = "";
            });
        });
    };

    // Apply to static cards
    apply3DEffect($$(".glass-card--3d"));

    // Expose globally to apply to dynamic cards later
    window.apply3DEffect = apply3DEffect;
}

// ============================================================
// Skill Management
// ============================================================
function addSkill() {
    const input = dom.skillInput.value.trim();
    if (!input) return;

    const skills = input.split(",").map((s) => s.trim()).filter(Boolean);

    skills.forEach((skill) => {
        if (!state.mustHaveSkills.includes(skill)) {
            state.mustHaveSkills.push(skill);
        }
    });

    dom.skillInput.value = "";
    renderSkillChips();
}

function removeSkill(skill) {
    state.mustHaveSkills = state.mustHaveSkills.filter((s) => s !== skill);
    renderSkillChips();
}

function renderSkillChips() {
    dom.skillChips.innerHTML = state.mustHaveSkills
        .map(
            (skill) => `
        <span class="skill-chip">
            ${escapeHtml(skill)}
            <button class="skill-chip__remove" onclick="removeSkill('${escapeHtml(skill)}')" title="Remove">&times;</button>
        </span>
    `
        )
        .join("");
}

// ============================================================
// Dropzone & File Upload
// ============================================================
function setupDropzone() {
    const dz = dom.dropzone;

    dz.addEventListener("click", () => dom.fileInput.click());

    dz.addEventListener("dragover", (e) => {
        e.preventDefault();
        dz.classList.add("dropzone--active");
    });

    dz.addEventListener("dragleave", () => {
        dz.classList.remove("dropzone--active");
    });

    dz.addEventListener("drop", (e) => {
        e.preventDefault();
        dz.classList.remove("dropzone--active");
        const files = Array.from(e.dataTransfer.files).filter((f) =>
            f.name.toLowerCase().endsWith(".pdf")
        );
        if (files.length) uploadFiles(files);
    });

    dom.fileInput.addEventListener("change", () => {
        const files = Array.from(dom.fileInput.files);
        if (files.length) uploadFiles(files);
        dom.fileInput.value = "";
    });
}

async function uploadFiles(files) {
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    try {
        const res = await fetch("/api/upload", { method: "POST", body: formData });
        const data = await res.json();

        if (data.uploaded?.length) {
            showToast(`Uploaded ${data.uploaded.length} resume(s)`, "success");
        }
        if (data.errors?.length) {
            showToast(data.errors.join(", "), "error");
        }

        loadResumes();
    } catch (err) {
        showToast("Upload failed: " + err.message, "error");
    }
}

// ============================================================
// Resume List
// ============================================================
async function loadResumes() {
    try {
        const res = await fetch("/api/resumes");
        if (!res.ok) return;
        const data = await res.json();

        dom.resumeCountText.textContent = `${data.count} resume${data.count !== 1 ? "s" : ""}`;

        if (data.resumes.length === 0) {
            dom.resumeList.innerHTML = "";
            return;
        }

        dom.resumeList.innerHTML = data.resumes
            .map(
                (r) => `
            <div class="resume-item">
                <div class="resume-item__name">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    <span title="${escapeHtml(r.filename)}">${escapeHtml(r.filename)}</span>
                </div>
                <span class="resume-item__size">${r.size_human}</span>
                <button class="resume-item__delete" onclick="deleteResume('${escapeHtml(r.filename)}')" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                </button>
            </div>
        `
            )
            .join("");
    } catch (err) {
        console.error("Failed to load resumes:", err);
    }
}

async function deleteResume(filename) {
    try {
        await fetch(`/api/resumes/${encodeURIComponent(filename)}`, {
            method: "DELETE",
        });
        showToast(`Deleted ${filename}`, "info");
        loadResumes();
    } catch (err) {
        showToast("Delete failed: " + err.message, "error");
    }
}

// ============================================================
// Analysis Pipeline
// ============================================================
async function runAnalysis() {
    const jdText = dom.jdInput.value.trim();
    if (!jdText) {
        showToast("Please enter a job description", "error");
        dom.jdInput.focus();
        return;
    }

    const useLLM = dom.llmToggle.checked && state.llmAvailable;

    // Show loading
    dom.analyzeBtn.disabled = true;
    dom.loadingSection.style.display = "flex";
    dom.resultsSection.style.display = "none";

    // Animate loading text
    const loadingSteps = useLLM
        ? [
            "Parsing resumes and extracting skills...",
            "Generating semantic embeddings...",
            "Building vector index...",
            "Computing composite scores...",
            "Running AI reasoning on top candidates...",
            "LLM evaluating candidate fit...",
            "Generating strengths & weaknesses...",
        ]
        : [
            "Parsing resumes and extracting skills...",
            "Generating semantic embeddings...",
            "Building vector index...",
            "Computing composite scores...",
            "Reranking candidates...",
        ];
    let stepIdx = 0;
    const loadingInterval = setInterval(() => {
        stepIdx = (stepIdx + 1) % loadingSteps.length;
        dom.loadingText.textContent = loadingSteps[stepIdx];
    }, 2000);

    try {
        const res = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                job_description: jdText,
                must_have_skills: state.mustHaveSkills,
                use_llm_rerank: useLLM,
            }),
        });

        const data = await res.json();

        clearInterval(loadingInterval);
        dom.loadingSection.style.display = "none";
        dom.analyzeBtn.disabled = false;

        if (!res.ok) {
            showToast(data.error || "Analysis failed", "error");
            return;
        }

        state.analysisData = data;
        state.results = data.results;
        renderResults(data);

    } catch (err) {
        clearInterval(loadingInterval);
        dom.loadingSection.style.display = "none";
        dom.analyzeBtn.disabled = false;
        showToast("Analysis failed: " + err.message, "error");
    }
}

async function downloadExcel() {
    if (!state.results || state.results.length === 0) {
        showToast("No results to download", "error");
        return;
    }

    try {
        const btn = dom.downloadExcelBtn;
        const originalText = btn.innerHTML;
        btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin" style="margin-right:6px; width:16px; height:16px;"><circle cx="12" cy="12" r="10" stroke-opacity="0.25"/><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Exporting...`;
        btn.disabled = true;

        const res = await fetch("/api/export/excel");
        if (!res.ok) {
            const data = await res.json();
            throw new Error(data.detail || data.error || "Export failed");
        }

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = "ai_recruiter_results.xlsx";
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

        showToast("Excel exported successfully", "success");
        
        btn.innerHTML = originalText;
        btn.disabled = false;
    } catch (err) {
        dom.downloadExcelBtn.disabled = false;
        dom.downloadExcelBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16" style="margin-right: 6px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> Export Excel`;
        showToast(err.message, "error");
    }
}

// ============================================================
// Render Results
// ============================================================
function renderResults(data) {
    dom.resultsSection.style.display = "block";

    // Smooth scroll to results
    setTimeout(() => {
        dom.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);

    // Stats
    $("#statCandidates").textContent = data.total_candidates;
    $("#statFiltered").textContent = data.filtered_count;
    $("#statSkills").textContent = data.jd_skills?.length || 0;
    $("#statTime").textContent = `${data.processing_time}s`;

    // LLM stat
    if (data.llm_used) {
        dom.statLLMContainer.style.display = "";
        $("#statLLM").textContent = "✓";
    } else {
        dom.statLLMContainer.style.display = "none";
    }

    // Detected JD skills
    dom.detectedSkills.innerHTML = (data.jd_skills || [])
        .map((s) => `<span class="detected-skill-tag">${escapeHtml(s)}</span>`)
        .join("");

    // Candidate cards
    dom.candidatesGrid.innerHTML = data.results
        .map((r, i) => renderCandidateCard(r, i))
        .join("");

    // Apply 3D effect to the new cards
    if (window.apply3DEffect) {
        window.apply3DEffect($$(".candidate-card"));
    }

    // Add click handlers for detail modals
    $$(".candidate-card").forEach((card) => {
        card.addEventListener("click", () => {
            const idx = parseInt(card.dataset.index);
            openModal(data.results[idx]);
        });
    });

    // Animate score bars
    setTimeout(() => {
        $$(".score-bar__fill").forEach((bar) => {
            bar.style.width = bar.dataset.width;
        });
    }, 100);
}

function renderCandidateCard(result, index) {
    const filtered = result.filtered;
    const llmEval = result.llm_evaluated;
    const rankClass = filtered
        ? "candidate-card__rank--filtered"
        : result.rank <= 3
        ? `candidate-card__rank--${result.rank}`
        : "candidate-card__rank--other";

    const rankText = filtered ? "✕" : `#${result.rank}`;
    
    // Use llm_score if available, otherwise final_score
    const displayScore = llmEval && result.llm_score !== undefined ? result.llm_score : result.final_score;
    const scorePercent = Math.round(displayScore * 100);

    // Decision badge (from LLM)
    const decisionHtml = llmEval && result.llm_decision
        ? `<span class="decision-badge decision-badge--${result.llm_decision.toLowerCase().replace(' ', '-')}">${result.llm_decision}</span>`
        : "";

    const missingHtml = (result.missing_skills || [])
        .slice(0, 3)
        .map((s) => `<span class="missing-skill-tag">${escapeHtml(s)}</span>`)
        .join("");

    const extraMissing =
        (result.missing_skills || []).length > 3
            ? `<span class="missing-skill-tag">+${result.missing_skills.length - 3} more</span>`
            : "";

    let explanationText = result.explanation;
    let extraAiHtml = "";
    if (llmEval) {
        if (result.llm_reasoning && result.llm_reasoning.length > 0) {
            explanationText = result.llm_reasoning[0];
        }
        const strengths = (result.llm_strengths || []).slice(0, 2)
            .map(s => `<span class="modal__tag modal__tag--matched" style="font-size:0.65rem; padding: 2px 6px; margin-right:4px;">+ ${escapeHtml(s)}</span>`).join("");
        if (strengths) {
            extraAiHtml = `<div style="margin-top:6px; margin-bottom: 6px;">${strengths}</div>`;
        }
    }

    return `
        <div class="candidate-card ${filtered ? "candidate-card--filtered" : ""}" data-index="${index}">
            <span class="candidate-card__rank ${rankClass}">${rankText}</span>
            <h3 class="candidate-card__name">${escapeHtml(result.filename.replace(".pdf", "").replace(/_/g, " "))}</h3>
            ${decisionHtml}
            <p class="candidate-card__explanation">${escapeHtml(explanationText)}</p>
            ${extraAiHtml}

            <div class="score-bar">
                <div class="score-bar__header">
                    <span class="score-bar__label">${llmEval ? "AI Score" : "Final Score"}</span>
                    <span class="score-bar__value">${filtered ? "0.000" : displayScore.toFixed(3)}</span>
                </div>
                <div class="score-bar__track">
                    <div class="score-bar__fill" style="width:0%" data-width="${scorePercent}%"></div>
                </div>
            </div>

            <div class="score-breakdown-mini">
                <div class="score-mini">
                    <div class="score-mini__value">${(result.semantic_score * 100).toFixed(0)}%</div>
                    <div class="score-mini__label">Semantic</div>
                </div>
                <div class="score-mini">
                    <div class="score-mini__value">${result.skill_match_pct.toFixed(0)}%</div>
                    <div class="score-mini__label">Skills</div>
                </div>
                <div class="score-mini">
                    <div class="score-mini__value">${result.experience_years || 0}yr</div>
                    <div class="score-mini__label">Experience</div>
                </div>
            </div>

            ${
                missingHtml || extraMissing
                    ? `<div class="missing-skills">${missingHtml}${extraMissing}</div>`
                    : ""
            }

            <div class="candidate-card__details">
                View details
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
            </div>
        </div>
    `;
}

// ============================================================
// Detail Modal (Enhanced with LLM data + Feedback)
// ============================================================
function openModal(result) {
    const filtered = result.filtered;
    const llmEval = result.llm_evaluated;

    const matchedHtml = (result.matched_skills || [])
        .map((s) => `<span class="modal__tag modal__tag--matched">${escapeHtml(s)}</span>`)
        .join("");

    const missingHtml = (result.missing_skills || [])
        .map((s) => `<span class="modal__tag modal__tag--missing">${escapeHtml(s)}</span>`)
        .join("");

    // LLM Assessment section
    let llmSection = "";
    if (llmEval) {
        const reasoningHtml = (result.llm_reasoning || [])
            .map((r) => `<li>${escapeHtml(r)}</li>`)
            .join("");
        const strengthsHtml = (result.llm_strengths || [])
            .map((s) => `<span class="modal__tag modal__tag--matched">${escapeHtml(s)}</span>`)
            .join("");
        const weaknessesHtml = (result.llm_weaknesses || [])
            .map((w) => `<span class="modal__tag modal__tag--missing">${escapeHtml(w)}</span>`)
            .join("");
        const riskHtml = (result.llm_risk_flags || [])
            .map((f) => `<span class="modal__tag modal__tag--risk">${escapeHtml(f)}</span>`)
            .join("");
        const questionsHtml = (result.llm_interview_questions || [])
            .map((q) => `<li>${escapeHtml(q)}</li>`)
            .join("");

        const decisionClass = result.llm_decision
            ? `decision-badge--${result.llm_decision.toLowerCase().replace(' ', '-')}`
            : "";

        llmSection = `
            <div class="modal__ai-section">
                <div class="modal__ai-header">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" width="18" height="18"><path d="M12 2a4 4 0 0 1 4 4c0 1.95-1.4 3.58-3.25 3.93L12 10v2"/><circle cx="12" cy="16" r="1"/></svg>
                    AI Assessment
                    ${result.llm_decision ? `<span class="decision-badge ${decisionClass}">${result.llm_decision}</span>` : ""}
                </div>

                ${reasoningHtml ? `
                <div class="modal__section">
                    <div class="modal__section-title">Reasoning</div>
                    <ul class="modal__reasoning">${reasoningHtml}</ul>
                </div>` : ""}

                ${strengthsHtml ? `
                <div class="modal__section">
                    <div class="modal__section-title">Strengths</div>
                    <div class="modal__tags">${strengthsHtml}</div>
                </div>` : ""}

                ${weaknessesHtml ? `
                <div class="modal__section">
                    <div class="modal__section-title">Weaknesses</div>
                    <div class="modal__tags">${weaknessesHtml}</div>
                </div>` : ""}

                ${riskHtml ? `
                <div class="modal__section">
                    <div class="modal__section-title">Risk Flags</div>
                    <div class="modal__tags">${riskHtml}</div>
                </div>` : ""}

                ${questionsHtml ? `
                <div class="modal__section">
                    <div class="modal__section-title">Interview Questions</div>
                    <ul class="modal__reasoning">${questionsHtml}</ul>
                </div>` : ""}
            </div>
        `;
    }

    dom.modalContent.innerHTML = `
        <h2 class="modal__title">${escapeHtml(result.filename.replace(".pdf", "").replace(/_/g, " "))}</h2>
        <p class="modal__subtitle">${escapeHtml(result.explanation)}</p>

        <div class="modal__score-big">
            <div class="score-value">${filtered ? "FILTERED" : result.final_score.toFixed(3)}</div>
            <div class="score-label">${filtered ? "Missing must-have skills" : "Composite Score"}</div>
        </div>

        <div class="modal__breakdown">
            <div class="breakdown-item">
                <div class="breakdown-item__value">${(result.semantic_score * 100).toFixed(0)}%</div>
                <div class="breakdown-item__label">Semantic</div>
                <div class="breakdown-item__weight">35%</div>
            </div>
            <div class="breakdown-item">
                <div class="breakdown-item__value">${(result.skill_score * 100).toFixed(0)}%</div>
                <div class="breakdown-item__label">Skills</div>
                <div class="breakdown-item__weight">25%</div>
            </div>
            <div class="breakdown-item">
                <div class="breakdown-item__value">${(result.experience_score * 100).toFixed(0)}%</div>
                <div class="breakdown-item__label">Experience</div>
                <div class="breakdown-item__weight">20%</div>
            </div>
            <div class="breakdown-item">
                <div class="breakdown-item__value">${(result.recency_score * 100).toFixed(0)}%</div>
                <div class="breakdown-item__label">Recency</div>
                <div class="breakdown-item__weight">10%</div>
            </div>
            <div class="breakdown-item">
                <div class="breakdown-item__value">${(result.keyword_score * 100).toFixed(0)}%</div>
                <div class="breakdown-item__label">Keywords</div>
                <div class="breakdown-item__weight">10%</div>
            </div>
        </div>

        <div class="modal__section">
            <div class="modal__section-title">Experience</div>
            <p style="color:var(--text-secondary); font-size:0.9rem;">
                ${result.experience_years || 0} year${result.experience_years !== 1 ? "s" : ""}
            </p>
        </div>

        ${
            matchedHtml
                ? `
            <div class="modal__section">
                <div class="modal__section-title">Matched Skills (${result.matched_skills?.length || 0})</div>
                <div class="modal__tags">${matchedHtml}</div>
            </div>
        `
                : ""
        }

        ${
            missingHtml
                ? `
            <div class="modal__section">
                <div class="modal__section-title">Missing Skills (${result.missing_skills?.length || 0})</div>
                <div class="modal__tags">${missingHtml}</div>
            </div>
        `
                : ""
        }

        ${llmSection}

        <!-- Feedback Actions -->
        <div class="modal__feedback" id="modalFeedback">
            <div class="modal__section-title">Your Decision</div>
            <div class="modal__feedback-btns">
                <button class="btn btn--shortlist" onclick="submitFeedback('${escapeHtml(result.filename)}', 'shortlisted')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16"><polyline points="20 6 9 17 4 12"/></svg>
                    Shortlist
                </button>
                <button class="btn btn--reject" onclick="submitFeedback('${escapeHtml(result.filename)}', 'rejected')">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" width="16" height="16"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    Reject
                </button>
            </div>
        </div>

        <!-- Interview Questions (on-demand) -->
        ${!llmEval ? `
        <div class="modal__section" style="margin-top:1rem;">
            <button class="btn btn--secondary" onclick="requestInterviewQuestions('${escapeHtml(result.filename)}', this)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                Generate Interview Questions
            </button>
            <div class="interview-questions-result" id="iq-${escapeHtml(result.filename).replace(/[^a-zA-Z0-9]/g, '_')}"></div>
        </div>` : ""}
    `;

    dom.modalOverlay.classList.add("modal-overlay--active");
    document.body.style.overflow = "hidden";
}

function closeModal() {
    dom.modalOverlay.classList.remove("modal-overlay--active");
    document.body.style.overflow = "";
}

// ============================================================
// Feedback
// ============================================================
async function submitFeedback(filename, action) {
    try {
        const res = await fetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename, action }),
        });
        const data = await res.json();

        if (data.recorded) {
            showToast(
                `${action === "shortlisted" ? "✅ Shortlisted" : "❌ Rejected"}: ${filename.replace(".pdf", "")}`,
                action === "shortlisted" ? "success" : "info"
            );

            // Update button states
            const feedbackDiv = $("#modalFeedback");
            if (feedbackDiv) {
                feedbackDiv.innerHTML = `
                    <div class="modal__section-title">Your Decision</div>
                    <div class="feedback-recorded">
                        ${action === "shortlisted"
                            ? '<span class="feedback-recorded--shortlisted">✅ Shortlisted</span>'
                            : '<span class="feedback-recorded--rejected">❌ Rejected</span>'}
                        <span class="feedback-recorded__count">${data.total_feedback} total decisions recorded</span>
                    </div>
                `;
            }
        }
    } catch (err) {
        showToast("Feedback failed: " + err.message, "error");
    }
}

// ============================================================
// Interview Questions (on-demand)
// ============================================================
async function requestInterviewQuestions(filename, btnElement) {
    btnElement.disabled = true;
    btnElement.textContent = "Generating...";

    const containerId = `iq-${filename.replace(/[^a-zA-Z0-9]/g, '_')}`;

    try {
        const res = await fetch(`/api/candidate/${encodeURIComponent(filename)}/interview-questions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ count: 5 }),
        });
        const data = await res.json();

        const container = document.getElementById(containerId);
        if (container && data.questions) {
            container.innerHTML = `
                <div class="modal__section" style="margin-top:0.75rem;">
                    <div class="modal__section-title">Interview Questions</div>
                    <ul class="modal__reasoning">
                        ${data.questions.map(q => `
                            <li>
                                <strong>${escapeHtml(q.question)}</strong>
                                ${q.purpose ? `<br><small style="color:var(--text-muted);">Purpose: ${escapeHtml(q.purpose)}</small>` : ""}
                            </li>
                        `).join("")}
                    </ul>
                </div>
            `;
        }

        btnElement.style.display = "none";
    } catch (err) {
        btnElement.disabled = false;
        btnElement.textContent = "Generate Interview Questions";
        showToast("Failed to generate questions: " + err.message, "error");
    }
}

// ============================================================
// Chat Drawer
// ============================================================
function toggleChat() {
    state.chatOpen = !state.chatOpen;
    dom.chatDrawer.classList.toggle("chat-drawer--open", state.chatOpen);
    if (state.chatOpen) {
        dom.chatInput.focus();
    }
}

async function sendChatMessage() {
    const message = dom.chatInput.value.trim();
    if (!message) return;

    // Add user message to chat
    appendChatMessage(message, "user");
    dom.chatInput.value = "";

    // Show typing indicator
    const typingId = appendChatMessage("Thinking...", "assistant", true);

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        const data = await res.json();

        // Remove typing indicator
        removeChatMessage(typingId);

        // Add response
        appendChatMessage(data.response || "No response", "assistant");

    } catch (err) {
        removeChatMessage(typingId);
        appendChatMessage(`Error: ${err.message}`, "assistant");
    }
}

let _chatMsgId = 0;
function appendChatMessage(text, role, isTyping = false) {
    const id = `chat-msg-${++_chatMsgId}`;
    const div = document.createElement("div");
    div.className = `chat-msg chat-msg--${role} ${isTyping ? "chat-msg--typing" : ""}`;
    div.id = id;

    // Simple markdown-like rendering for bold
    let rendered = escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");

    div.innerHTML = `<div class="chat-msg__content">${rendered}</div>`;
    dom.chatMessages.appendChild(div);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    return id;
}

function removeChatMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ============================================================
// Toast Notifications
// ============================================================
function showToast(message, type = "info") {
    let container = $(".toast-container");
    if (!container) {
        container = document.createElement("div");
        container.className = "toast-container";
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => toast.remove(), 4000);
}

// ============================================================
// Utilities
// ============================================================
function escapeHtml(str) {
    if (!str) return "";
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// Make functions globally accessible
window.removeSkill = removeSkill;
window.deleteResume = deleteResume;
window.submitFeedback = submitFeedback;
window.requestInterviewQuestions = requestInterviewQuestions;
