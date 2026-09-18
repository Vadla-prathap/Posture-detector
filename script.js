/* =========================================================
   AI MOVEMENT COACH — FRONTEND CONTROLLER
   (original app logic preserved; NEW sections are marked)
========================================================= */

const state = {
    currentPage: "dashboard",
    previousPage: "practice",
    selectedMode: "dance",
    selectedCamera: "front",
    countdown: 5,

    cameraStream: null,
    mediaRecorder: null,
    recordedChunks: [],
    recordingBlob: null,
    recordingURL: null,
    referenceURL: null,
    referenceFile: null,
    /* ---- NEW: reference video can now come from a pasted link, not
       just a file upload. sourceType tracks which playback path is
       active so every downstream function (studio preview, recording
       sync, Gemini upload) knows how to handle it. ---- */
    referenceSourceType: "file", // "file" | "url" | "youtube" | "instagram"
    youtubeVideoId: null,
    studioYoutubePlayer: null,

    isRecording: false,
    isCountingDown: false,
    practiceStartTime: null,
    timerInterval: null,
    countdownTimer: null,

    currentSession: null,
    history: [],
    sessions: 0,
    bestScore: 0,
    totalIssues: 0,

    feedbackLanguage: "en-IN",
    currentFeedbackText: "",
    currentMistakes: [],
    currentTranslations: null,

    seniorMode: false,
    voiceLanguage: "en-IN",
    coachEnabled: true,

    planGoal: "general",
    planLevel: "beginner",
    savedPlan: null,

    nutritionHistory: [],

    chatBusy: false,
    speechRecognition: null,
    isListening: false,

    poseEnabled: true,

    /* ---- NEW: automatic exercise recognition ---- */
    autoDetectEnabled: true,
    detectedExercise: null,
    detectionConfidence: 0,
    activeExercise: null,
    geminiAvailable: null,

    /* ---- NEW: rep counting / form scoring for the current set ---- */
    sessionMetrics: {
        reps: 0,
        correctReps: 0,
        corrections: 0,
        formScoreSamples: [],
        jointSamples: [],
        issueTallies: {},
        recognitionConfidence: 0
    },

    /* ---- NEW: achievements (gamification) ---- */
    achievements: []
};


/* =========================================================
   POSE TRACKING (MediaPipe Tasks Vision) — runs on-device
========================================================= */

const POSE_VISION_MODULE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const POSE_WASM_BASE_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const POSE_LANDMARKS = {
    NOSE: 0,
    LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
    LEFT_WRIST: 15, RIGHT_WRIST: 16,
    LEFT_HIP: 23, RIGHT_HIP: 24,
    LEFT_KNEE: 25, RIGHT_KNEE: 26,
    LEFT_ANKLE: 27, RIGHT_ANKLE: 28
};

const poseEngine = {
    Vision: null,
    landmarker: null,
    loading: false,
    failed: false,
    loopHandle: null,
    lastDetectTime: 0,
    lastFrameLandmarks: null,
    lastFrameTime: 0,
    lastCueMessage: "",
    lastCueTime: -Infinity,
    cueCooldowns: {},
    /* NEW: rolling buffer of computed joint angles used for exercise recognition */
    angleBuffer: [],
    lastClassifyTime: 0,
    drawingUtils: null,
    smoothedAngles: null,
    lastHudKey: "",
    lastHudTime: 0,
    noPoseFrames: 0,
    recognition: {
        pending: null,
        pendingHits: 0,
        locked: null,
        lockedConfidence: 0,
        lowHits: 0
    }
};

const POSE_DETECT_INTERVAL_MS = 90;
const POSE_MIN_CUE_GAP_MS = 2200;
const POSE_SAME_CUE_GAP_MS = 6000;
const ANGLE_BUFFER_WINDOW_MS = 1800;
const CLASSIFY_INTERVAL_MS = 300;
const CONFIDENCE_THRESHOLD = 52;
const RECOGNITION_LOCK_HITS = 2;
const RECOGNITION_HOLD_CONFIDENCE = 30;
const RECOGNITION_SWITCH_MARGIN = 14;
const HUD_UPDATE_INTERVAL_MS = 140;
const REP_COOLDOWN_MS = 480;
const FORM_SMOOTH = 0.28;


/* =========================================================
   NEW: REP COUNTING CONFIG (per recognized/selected exercise)
   direction "decrease": rep starts at rest (high metric value),
   dips below activeThreshold, returns above restThreshold.
   direction "increase": inverse (used for overhead presses).
========================================================= */

const REP_CONFIG = {

    squat: {
        metric: a => a.kneeAvg,
        direction: "decrease",
        activeThreshold: 112,
        restThreshold: 152,
        minRange: 32,
        formCheck: a => (a.backAngle == null || a.backAngle < 36) && (a.kneeValgusMax == null || a.kneeValgusMax < 0.07)
    },

    lunge: {
        metric: a => a.frontKneeAngle != null ? a.frontKneeAngle : a.kneeAvg,
        direction: "decrease",
        activeThreshold: 118,
        restThreshold: 152,
        minRange: 28,
        formCheck: a => (a.backAngle == null || a.backAngle < 36) && (a.ankleYDiff == null || a.ankleYDiff > 0.04)
    },

    pushup: {
        metric: a => a.elbowAvg,
        direction: "decrease",
        activeThreshold: 100,
        restThreshold: 148,
        minRange: 35,
        formCheck: a => {
            const bodyAngle = angleAt(a.shoulderMid, a.hipMid, a.ankleMid);
            return bodyAngle == null || bodyAngle > 152;
        }
    },

    bicepcurl: {
        metric: a => a.elbowAvg,
        direction: "decrease",
        activeThreshold: 72,
        restThreshold: 148,
        minRange: 40,
        formCheck: a => a.elbowNearTorso
    },

    shoulderpress: {
        metric: a => a.pressHeight,
        direction: "increase",
        activeThreshold: 0.14,
        restThreshold: 0.05,
        minRange: 0.08,
        formCheck: a => a.backAngle == null || a.backAngle < 24
    }

};

const repState = {
    exercise: null,
    phase: "rest",
    count: 0,
    correctCount: 0,
    lastTransitionTime: 0,
    extremum: null,
    startValue: null,
    formOkDuringRep: true
};

/* =========================================================
   NEW: REP COUNTING STATE MACHINE
   Now enforces cfg.minRange (it was defined per-exercise but
   never actually read before) — a rep only counts if the
   tracked joint travelled far enough between the rest baseline
   and the deepest/highest point reached, so a small, noisy
   wobble around the threshold can no longer register as a rep.
========================================================= */

function updateRepCounting(exercise, angles, now) {
    const cfg = REP_CONFIG[exercise];
    if (!cfg) return null;

    const value = cfg.metric(angles);
    if (value == null) return null;

    if (repState.exercise !== exercise) {
        repState.exercise = exercise;
        repState.phase = "rest";
        repState.count = 0;
        repState.correctCount = 0;
        repState.extremum = null;
    }

    if (now - repState.lastTransitionTime < 300) return null;

    const isActive = cfg.direction === "decrease" ? value < cfg.activeThreshold : value > cfg.activeThreshold;
    const isRest = cfg.direction === "decrease" ? value > cfg.restThreshold : value < cfg.restThreshold;

    if (repState.phase === "rest" && isActive) {
        repState.phase = "active";
        repState.lastTransitionTime = now;
        repState.extremum = value;
        return null;
    }

    if (repState.phase === "active") {
        // Track the deepest/highest point reached during this rep attempt.
        repState.extremum = cfg.direction === "decrease"
            ? Math.min(repState.extremum ?? value, value)
            : Math.max(repState.extremum ?? value, value);

        if (isRest) {
            repState.phase = "rest";
            repState.lastTransitionTime = now;

            const range = Math.abs(cfg.restThreshold - repState.extremum);

            if (range < cfg.minRange) {
                // Didn't travel far enough to count as a genuine rep —
                // likely noise or an incomplete movement. Don't count it.
                repState.extremum = null;
                return { repCompleted: false, tooShallow: true };
            }

            repState.count++;

            const formOk = cfg.formCheck ? cfg.formCheck(angles) : true;
            if (formOk) repState.correctCount++;

            repState.extremum = null;

            return { repCompleted: true, correct: formOk, count: repState.count };
        }
    }

    return null;
}

const LIVE_CUE_I18N = {
    "Keep your knees aligned.": { te: "మీ మోకాళ్లను సరైన రేఖలో ఉంచండి.", hi: "घुटनों को सीध में रखें।" },
    "Straighten your back.": { te: "మీ వీపును నిటారుగా ఉంచండి.", hi: "पीठ सीधी रखें।" },
    "Go slightly deeper.": { te: "కొంచెం లోతుగా వెళ్లండి.", hi: "थोड़ा और नीचे जाएं।" },
    "Good rep.": { te: "మంచి రెప్.", hi: "अच्छा रेप।" },
    "Keep your back straight — engage your core.": { te: "వీపు నిటారుగా ఉంచి కోర్‌ను బిగించండి.", hi: "पीठ सीधी रखें और कोर टाइट रखें।" },
    "Keep your left elbow tucked in close to your body.": { te: "ఎడమ మోచేయి శరీరానికి దగ్గరగా ఉంచండి.", hi: "बाएं एल्बो को शरीर के पास रखें।" },
    "Keep your right elbow tucked in close to your body.": { te: "కుడి మోచేయి శరీరానికి దగ్గరగా ఉంచండి.", hi: "दाएं एल्बो को शरीर के पास रखें।" },
    "Avoid arching your back — keep your core engaged.": { te: "వీపు వంచకుండా కోర్‌ను బిగించండి.", hi: "पीठ न झुकाएं, कोर टाइट रखें।" },
    "Keep your shoulders aligned.": { te: "భుజాలను సమానంగా ఉంచండి.", hi: "कंधों को संतुलित रखें।" },
    "Keep your shoulders level.": { te: "భుజాలు సమాంతరంగా ఉంచండి.", hi: "कंधे एक स्तर पर रखें।" },
    "Keep your head aligned over your shoulders.": { te: "తలను భుజాలపై సమం చేయండి.", hi: "सिर को कंधों के ऊपर संरेखित रखें।" },
    "Slow down and control the movement.": { te: "నెమ్మదిగా, నియంత్రణతో కదలండి.", hi: "धीमे और नियंत्रण से चलें।" },
    "Keep your left hand up — match your right arm's height.": { te: "ఎడమ చేతిని కుడి చేతి ఎత్తుకు తీసుకురండి.", hi: "बाएं हाथ को दाएं हाथ की ऊँचाई पर रखें।" },
    "Keep your right hand up — match your left arm's height.": { te: "కుడి చేతిని ఎడమ చేతి ఎత్తుకు తీసుకురండి.", hi: "दाएं हाथ को बाएं हाथ की ऊँचाई पर रखें।" },
    "Keep your hips in line.": { te: "నడుమును ఒక లైన్‌లో ఉంచండి.", hi: "कमर को एक रेखा में रखें।" },
    "Keep both arms even.": { te: "రెండు చేతులను సమానంగా ఉంచండి.", hi: "दोनों बाँहें समान रखें।" }
};

function localizeCue(message) {
    if (!message) return "";
    const lang = (state.voiceLanguage || "en-IN").startsWith("te") ? "te" : (state.voiceLanguage || "").startsWith("hi") ? "hi" : "en";
    if (lang === "en") return message;
    return LIVE_CUE_I18N[message]?.[lang] || message;
}


/* =========================================================
   NEW: UI PHRASE TRANSLATIONS
   Covers the client-built voice/summary sentences that are NOT
   part of Gemini's own multilingual JSON response (session-score
   narration, the short post-analysis spoken summary, and the
   offline fallback used only if the coach-recommendation network
   call itself fails). These were previously hardcoded English
   strings just tagged with a different TTS "lang" attribute —
   the language selector had no effect on the actual words spoken.
========================================================= */

const UI_PHRASES = {
    "en-IN": {
        scoreIs: (score) => `Your movement score is ${score} percent.`,
        excellentNoMistakes: "Excellent work. No major movement mistakes were detected.",
        foundAreas: (count) => `I found ${count} ${count === 1 ? "area" : "areas"} to improve.`,
        issueN: (n) => `Issue ${n}.`,
        toFixIt: "To fix it:",
        greatSession: "Great session. Your movement matched the reference closely.",
        goodSession: "Good session. A couple of small corrections to work on.",
        sessionComplete: "Session complete. Let's review a few corrections.",
        topTip: (title) => `Top tip: ${title}.`,
        solidSession: "Solid session \u2014 your form held up well.",
        goodEffort: "Good effort \u2014 a few form corrections will help next time.",
        keepPracticing: "Keep practicing this exercise at a steady pace and focus on the highlighted corrections above.",
    },
    "te-IN": {
        scoreIs: (score) => `మీ కదలిక స్కోరు ${score} శాతం.`,
        excellentNoMistakes: "అద్భుతం. పెద్ద తప్పులు ఏవీ కనిపించలేదు.",
        foundAreas: (count) => `మెరుగుపరచడానికి ${count} అంశాలు కనిపించాయి.`,
        issueN: (n) => `సమస్య ${n}.`,
        toFixIt: "దీన్ని సరిదిద్దడానికి:",
        greatSession: "అద్భుతమైన సెషన్. మీ కదలిక రిఫరెన్స్‌కు దగ్గరగా సరిపోయింది.",
        goodSession: "మంచి సెషన్. కొన్ని చిన్న సవరణలు అవసరం.",
        sessionComplete: "సెషన్ పూర్తయింది. కొన్ని సవరణలను పరిశీలిద్దాం.",
        topTip: (title) => `ముఖ్య సూచన: ${title}.`,
        solidSession: "మంచి సెషన్ \u2014 మీ ఫారమ్ బాగా నిలబెట్టుకున్నారు.",
        goodEffort: "మంచి ప్రయత్నం \u2014 కొన్ని ఫారమ్ సవరణలు తదుపరిసారి సహాయపడతాయి.",
        keepPracticing: "ఈ వ్యాయామాన్ని స్థిరమైన వేగంతో సాధన చేస్తూ, పైన హైలైట్ చేసిన సవరణలపై దృష్టి పెట్టండి.",
    },
    "hi-IN": {
        scoreIs: (score) => `आपका मूवमेंट स्कोर ${score} प्रतिशत है।`,
        excellentNoMistakes: "बहुत बढ़िया। कोई बड़ी गलती नहीं मिली।",
        foundAreas: (count) => `सुधार के लिए ${count} क्षेत्र मिले।`,
        issueN: (n) => `समस्या ${n}।`,
        toFixIt: "इसे ठीक करने के लिए:",
        greatSession: "शानदार सेशन। आपकी मूवमेंट रेफरेंस से काफी मिलती-जुलती रही।",
        goodSession: "अच्छा सेशन। कुछ छोटी सुधार आवश्यक हैं।",
        sessionComplete: "सेशन पूरा हुआ। चलिए कुछ सुधारों पर नज़र डालते हैं।",
        topTip: (title) => `मुख्य सुझाव: ${title}।`,
        solidSession: "बढ़िया सेशन \u2014 आपका फॉर्म अच्छा बना रहा।",
        goodEffort: "अच्छा प्रयास \u2014 कुछ फॉर्म सुधार अगली बार मदद करेंगे।",
        keepPracticing: "इस एक्सरसाइज़ को स्थिर गति से अभ्यास करें और ऊपर बताए गए सुधारों पर ध्यान दें।",
    },
};

function phrase(key, language, ...args) {
    const table = UI_PHRASES[language] || UI_PHRASES["en-IN"];
    const value = table[key] ?? UI_PHRASES["en-IN"][key];
    return typeof value === "function" ? value(...args) : value;
}


/* =========================================================
   MODE DATA
========================================================= */

const modeData = {

    dance: {
        title: "Dance", icon: "💃",
        description: "Compare your movement with a teacher or choreography reference.",
        reference: true, trackable: false,
        doList: ["Warm up for a minute before starting.", "Keep your movements relaxed and controlled.", "Watch the reference video once before recording."],
        dontList: ["Don't rush through the steps.", "Don't dance on a slippery or cluttered floor."]
    },

    gym: {
        title: "Gym", icon: "🏋️",
        description: "Analyze exercise posture and movement technique.",
        reference: true, trackable: true,
        doList: ["Keep your back straight during lifts.", "Breathe out on effort, in on release.", "Use light or no weight while learning form."],
        dontList: ["Don't lock your joints forcefully.", "Don't hold your breath through the whole set."]
    },

    yoga: {
        title: "Yoga", icon: "🧘",
        description: "Monitor pose alignment and controlled movement.",
        reference: true, trackable: false,
        doList: ["Move slowly into each pose.", "Keep your breathing steady and calm.", "Use a mat or non-slip surface."],
        dontList: ["Don't force a stretch beyond comfort.", "Don't hold your breath during poses."]
    },

    fitness: {
        title: "Fitness", icon: "⚡",
        description: "Improve movement technique and consistency.",
        reference: true, trackable: true,
        doList: ["Start with a light warm-up.", "Keep movements smooth and controlled.", "Rest between rounds if needed."],
        dontList: ["Don't move too quickly.", "Don't skip rest when you feel tired."]
    },

    exercise: {
        title: "General Exercise", icon: "🏃",
        description: "Monitor general movement and exercise technique.",
        reference: true, trackable: true,
        doList: ["Keep a steady, comfortable pace.", "Stand on a flat, clear surface.", "Keep your knees slightly soft, not locked."],
        dontList: ["Don't push through sharp pain.", "Don't exercise on an uneven surface."]
    },

    wellness: {
        title: "Healthcare / Wellness", icon: "♥",
        description: "Improve posture and movement awareness.",
        reference: true, trackable: false,
        doList: ["Keep your back straight while standing or sitting.", "Move gently and at your own pace.", "Keep a chair or wall nearby for balance if needed."],
        dontList: ["Don't move quickly or twist suddenly.", "Don't ignore dizziness or discomfort."]
    }

};

const EXERCISE_LABELS = {
    squat: "Squat", lunge: "Lunge", pushup: "Push-up",
    bicepcurl: "Bicep Curl", shoulderpress: "Shoulder Press",
    dance: "Dance Movement", standing: "Standing Posture"
};


/* =========================================================
   DOM HELPERS
========================================================= */

const $ = id => document.getElementById(id);
const $$ = selector => document.querySelectorAll(selector);


/* =========================================================
   INITIALIZATION
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* ---- VERSION STAMP: if you don't see this exact line in your
       browser console when the page loads, your browser is running
       a CACHED copy of the old script.js, not the file you just
       replaced. That would explain every fix appearing to "not work" —
       the new code was never actually running. Hard-refresh (Ctrl+Shift+R
       on Windows/Linux, Cmd+Shift+R on Mac) or open DevTools → Network
       tab → check "Disable cache" while testing. ---- */
    console.log("%c[MovementCoach] script.js BUILD 2024-skeleton-scale-fix loaded at " + new Date().toLocaleTimeString(), "background:#7c3aed;color:#fff;padding:4px 8px;border-radius:4px;font-weight:bold;");

    loadHistory();
    loadSeniorMode();
    loadVoiceLanguage();
    loadSavedPlan();
    loadNutritionHistory();
    loadAchievements();

    setupNavigation();
    setupModeButtons();
    setupSetupControls();
    setupUrlReferenceUI();
    setupStudioControls();
    setupMobileMenu();
    setupFeedbackVoice();
    setupSeniorMode();
    setupFitnessPlan();
    setupPrivacyControls();
    setupChatWidget();
    setupReportDownload();

    updateProgressUI();
    updateMistakeBadge();
    checkBackend();

    showPage("dashboard");
});


/* =========================================================
   NAVIGATION
========================================================= */

function setupNavigation() {
    $$(".nav-item[data-page]").forEach(button => {
        button.addEventListener("click", () => {
            showPage(button.dataset.page);
            closeMobileMenu();
        });
    });

    $$("[data-page-target]").forEach(button => {
        button.addEventListener("click", () => showPage(button.dataset.pageTarget));
    });
}

function showPage(page) {
    if (!$(page)) return;

    if (state.currentPage === "studio" && page !== "studio") {
        if (state.isRecording) {
            stopPractice(false);
        } else {
            cleanupStudio();
        }
    }

    $$(".page").forEach(section => section.classList.remove("active-page"));
    $(page).classList.add("active-page");

    $$(".nav-item[data-page]").forEach(button => {
        button.classList.toggle("active", button.dataset.page === page);
    });

    state.currentPage = page;
    window.scrollTo({ top: 0, behavior: "smooth" });

    if (page === "progress") updateProgressUI();
    if (page === "mistakes") renderHistoryMistakes();
}


/* =========================================================
   MOBILE
========================================================= */

function setupMobileMenu() {
    const menuBtn = $("menuBtn");
    if (!menuBtn) return;
    menuBtn.addEventListener("click", () => $("sidebar").classList.toggle("open"));
}

function closeMobileMenu() {
    $("sidebar")?.classList.remove("open");
}


/* =========================================================
   MODE
========================================================= */

function setupModeButtons() {
    $$(".mode-card[data-mode]").forEach(button => {
        button.addEventListener("click", () => selectMode(button.dataset.mode));
    });
}

function selectMode(mode) {
    if (!modeData[mode]) return;

    state.selectedMode = mode;
    const data = modeData[mode];

    $("selectedModeIcon").textContent = data.icon;
    $("selectedModeName").textContent = data.title;
    $("modeTitle").textContent = data.title;
    $("modeDescription").textContent = data.description;
    $("modeEyebrow").textContent = data.title.toUpperCase() + " TRAINING";
    $("studioModeTitle").textContent = data.title + " Training";

    $("referenceSetupCard").classList.toggle("hidden", !data.reference);

    const badge = $("referenceRequiredBadge");
    if (badge) {
        const required = mode === "dance";
        badge.textContent = required ? "REQUIRED" : "OPTIONAL";
        badge.classList.toggle("optional-badge", !required);
    }

    renderSeniorTips(data);
    showPage("modeSetup");
}


/* =========================================================
   SENIOR MODE: DO / DON'T TIPS
========================================================= */

function renderSeniorTips(data) {
    const card = $("seniorTipsCard");
    if (!card) return;

    if (!state.seniorMode) {
        card.classList.add("hidden");
        return;
    }

    $("seniorDoList").innerHTML = (data.doList || []).map(item => `<li>${escapeHTML(item)}</li>`).join("");
    $("seniorDontList").innerHTML = (data.dontList || []).map(item => `<li>${escapeHTML(item)}</li>`).join("");
    card.classList.remove("hidden");
}


/* =========================================================
   SETUP
========================================================= */

function setupSetupControls() {

    $("browseBtn").addEventListener("click", () => $("videoInput").click());

    $("videoInput").addEventListener("change", event => {
        const file = event.target.files[0];
        if (file) handleReferenceVideo(file);
    });

    $("removeVideo").addEventListener("click", removeReferenceVideo);

    $$("[data-camera]").forEach(button => {
        button.addEventListener("click", () => {
            $$("[data-camera]").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            state.selectedCamera = button.dataset.camera;
        });
    });

    $$("[data-countdown]").forEach(button => {
        button.addEventListener("click", () => {
            $$("[data-countdown]").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            state.countdown = Number(button.dataset.countdown);
        });
    });

    $$("[data-voice-lang]").forEach(button => {
        button.addEventListener("click", () => {
            $$("[data-voice-lang]").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            setVoiceLanguage(button.dataset.voiceLang);
        });
    });

    $$("[data-coach]").forEach(button => {
        button.addEventListener("click", () => {
            $$("[data-coach]").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            state.coachEnabled = button.dataset.coach === "on";
        });
    });

    $$("[data-pose]").forEach(button => {
        button.addEventListener("click", () => {
            $$("[data-pose]").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            state.poseEnabled = button.dataset.pose === "on";
        });
    });

    /* ---- NEW: automatic exercise recognition toggle ---- */
    $$("[data-autodetect]").forEach(button => {
        button.addEventListener("click", () => {
            $$("[data-autodetect]").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            state.autoDetectEnabled = button.dataset.autodetect === "on";

            const exerciseSelect = $("exerciseSelect");
            if (exerciseSelect) exerciseSelect.disabled = state.autoDetectEnabled;
        });
    });

    $("openStudio").addEventListener("click", openStudio);
    $("analysisPracticeAgain").addEventListener("click", practiceAgain);
}


/* =========================================================
   REFERENCE VIDEO
========================================================= */

function handleReferenceVideo(file) {
    if (!file.type.startsWith("video/")) {
        showToast("Please select a video file.");
        return;
    }

    if (state.referenceURL && state.referenceSourceType === "file") URL.revokeObjectURL(state.referenceURL);
    resetReferenceEmbeds();

    state.referenceFile = file;
    state.referenceURL = URL.createObjectURL(file);
    state.referenceSourceType = "file";
    state.youtubeVideoId = null;

    const video = $("referenceVideo");
    video.classList.remove("hidden");
    video.src = state.referenceURL;
    $("videoName").textContent = file.name;
    $("videoDuration").textContent = "Ready";
    $("uploadZone").classList.add("hidden");
    $("videoPreview").classList.remove("hidden");

    video.addEventListener("loadedmetadata", updateReferenceDuration, { once: true });
    if ($("referenceUrlInput")) $("referenceUrlInput").value = "";
    showToast("Reference video loaded.");
}

function updateReferenceDuration() {
    const video = $("referenceVideo");
    $("videoDuration").textContent = Number.isFinite(video.duration) ? formatTime(video.duration) : "Ready";
}

/* =========================================================
   NEW: REFERENCE VIDEO BY LINK (YouTube / Instagram / direct URL)
   Browsers cannot play a YouTube or Instagram page URL through a
   native <video> tag — those platforms only allow playback through
   their own embed players, and neither exposes a way to download
   the underlying file (Instagram has no public playback-control API
   at all; scraping YouTube's stream violates its ToS and breaks
   constantly). So this is built honestly in three tiers:
     - direct video file URL (.mp4 etc.)  -> full support, identical
       to file upload, including Gemini comparison (fetched server-side)
     - YouTube link                       -> embedded + controllable
       for live visual reference only; NOT sent to Gemini
     - Instagram link                     -> preview embed only, no
       playback control, NOT sent to Gemini
========================================================= */

const YOUTUBE_URL_RE = /(?:youtube\.com\/(?:watch\?v=|shorts\/|embed\/)|youtu\.be\/)([A-Za-z0-9_-]{6,})/i;
const INSTAGRAM_URL_RE = /instagram\.com\/(?:reel|p|tv)\/([A-Za-z0-9_-]+)/i;
const DIRECT_VIDEO_RE = /\.(mp4|webm|ogg|ogv|mov|m4v)(\?.*)?$/i;

function classifyReferenceUrl(rawUrl) {
    let url;
    try {
        url = new URL(rawUrl.trim());
    } catch (error) {
        return { type: "invalid" };
    }
    if (!/^https?:$/.test(url.protocol)) return { type: "invalid" };

    const youtubeMatch = rawUrl.match(YOUTUBE_URL_RE);
    if (youtubeMatch) return { type: "youtube", id: youtubeMatch[1], url: rawUrl.trim() };

    const instagramMatch = rawUrl.match(INSTAGRAM_URL_RE);
    if (instagramMatch) return { type: "instagram", id: instagramMatch[1], url: rawUrl.trim() };

    if (DIRECT_VIDEO_RE.test(url.pathname)) return { type: "direct", url: rawUrl.trim() };

    // Unknown host/path shape — still allow it as a best-effort direct link,
    // since some CDNs serve video without a file-extension in the path.
    return { type: "direct", url: rawUrl.trim() };
}

function setupUrlReferenceUI() {
    $("loadUrlBtn")?.addEventListener("click", () => {
        const raw = $("referenceUrlInput")?.value || "";
        if (!raw.trim()) {
            showToast("Paste a video link first.");
            return;
        }
        loadReferenceFromUrl(raw.trim());
    });

    $("referenceUrlInput")?.addEventListener("keydown", event => {
        if (event.key === "Enter") {
            event.preventDefault();
            $("loadUrlBtn")?.click();
        }
    });
}

function resetReferenceEmbeds() {
    const embedWrap = $("referenceEmbedWrap");
    const embedFrame = $("referenceEmbedFrame");
    if (embedFrame) embedFrame.src = "";
    embedWrap?.classList.add("hidden");
    $("referenceVideo")?.classList.remove("hidden");
}

function loadReferenceFromUrl(rawUrl) {
    const classified = classifyReferenceUrl(rawUrl);

    if (classified.type === "invalid") {
        showToast("That doesn't look like a valid video link.");
        return;
    }

    // Clear any existing file-based reference first (this REPLACES the
    // current reference, it doesn't stack with file upload).
    if (state.referenceURL && state.referenceSourceType === "file") {
        URL.revokeObjectURL(state.referenceURL);
    }
    state.referenceFile = null;
    resetReferenceEmbeds();

    if (classified.type === "youtube") {
        state.referenceSourceType = "youtube";
        state.youtubeVideoId = classified.id;
        state.referenceURL = classified.url;

        $("referenceVideo").classList.add("hidden");
        const embedWrap = $("referenceEmbedWrap");
        const embedFrame = $("referenceEmbedFrame");
        embedFrame.src = `https://www.youtube.com/embed/${classified.id}?enablejsapi=1&playsinline=1`;
        embedWrap.classList.remove("hidden");

        $("videoName").textContent = "YouTube reference";
        $("videoDuration").textContent = "Preview only \u2014 not sent to Gemini";
        $("uploadZone").classList.add("hidden");
        $("videoPreview").classList.remove("hidden");
        showToast("YouTube reference loaded. It will play during practice but won't be included in the Gemini comparison.");

    } else if (classified.type === "instagram") {
        state.referenceSourceType = "instagram";
        state.referenceURL = classified.url;
        state.youtubeVideoId = null;

        $("referenceVideo").classList.add("hidden");
        const embedWrap = $("referenceEmbedWrap");
        const embedFrame = $("referenceEmbedFrame");
        // Instagram has no public embeddable player URL without their
        // widget script + approval; show a clear, honest message instead
        // of a broken iframe.
        embedFrame.src = "";
        embedWrap.classList.add("hidden");

        $("videoName").textContent = "Instagram reference";
        $("videoDuration").textContent = "Link saved \u2014 preview unavailable";
        $("uploadZone").classList.add("hidden");
        $("videoPreview").classList.remove("hidden");
        showToast("Instagram links can't be embedded or auto-played in-browser. The link is saved for your own reference, but won't play here or reach Gemini \u2014 consider downloading the clip and uploading it as a file instead.");

    } else {
        // Direct video file URL — behaves exactly like a file upload.
        state.referenceSourceType = "url";
        state.referenceURL = classified.url;
        state.youtubeVideoId = null;

        const video = $("referenceVideo");
        video.classList.remove("hidden");
        video.src = classified.url;
        $("videoName").textContent = "Linked reference video";
        $("videoDuration").textContent = "Loading\u2026";
        $("uploadZone").classList.add("hidden");
        $("videoPreview").classList.remove("hidden");
        video.addEventListener("loadedmetadata", updateReferenceDuration, { once: true });
        video.addEventListener("error", () => {
            showToast("Couldn't load that video link \u2014 check the URL is a direct, publicly accessible video file.");
        }, { once: true });
        showToast("Reference video link loaded.");
    }

    if ($("referenceUrlInput")) $("referenceUrlInput").value = "";
}

function removeReferenceVideo() {
    const video = $("referenceVideo");
    video.pause();
    video.removeAttribute("src");
    video.load();

    if (state.referenceURL && state.referenceSourceType === "file") {
        URL.revokeObjectURL(state.referenceURL);
    }

    state.referenceURL = null;
    state.referenceFile = null;
    state.referenceSourceType = "file";
    state.youtubeVideoId = null;

    resetReferenceEmbeds();

    $("videoInput").value = "";
    if ($("referenceUrlInput")) $("referenceUrlInput").value = "";
    $("videoPreview").classList.add("hidden");
    $("uploadZone").classList.remove("hidden");
    showToast("Reference video removed.");
}


/* =========================================================
   OPEN STUDIO
========================================================= */

async function openStudio() {
    if (state.selectedMode === "dance" && !state.referenceURL) {
        showToast("Add a reference video first (upload a file or paste a link).");
        return;
    }

    showPage("studio");
    state.previousPage = "modeSetup";
    resetStudioUI();

    try {
        await startCamera();
        prepareReferenceForStudio();
        const hero = document.querySelector(".practice-stage");
        hero?.classList.toggle("camera-hero", !state.referenceURL);
        $("referencePanel")?.classList.toggle("hidden", !state.referenceURL);
    } catch (error) {
        console.error(error);
        const denied = error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError";
        showToast(denied
            ? "Camera permission denied. Enable camera in your browser settings, then try again."
            : "Camera could not be started. Check browser permission.");
        $("cameraStatus").textContent = "ERROR";
        $("liveFeedback").textContent = denied
            ? "Camera blocked — allow camera access and reopen training."
            : "Camera unavailable.";
    }
}


/* =========================================================
   CAMERA
========================================================= */

async function startCamera() {
    stopCameraOnly();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API unavailable");
    }

    const facingMode = state.selectedCamera === "back" ? "environment" : "user";

    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: true
    });

    state.cameraStream = stream;

    const cameraVideo = $("cameraVideo");
    cameraVideo.srcObject = stream;
    cameraVideo.muted = true;
    await cameraVideo.play();

    /* ---- FIX: the "mirror" class (CSS scaleX(-1)) was hardcoded in the
       HTML on both the camera video AND the pose canvas, so the REAR
       camera was also flipped — showing the world backwards, unlike a
       normal phone camera. A mirror only makes sense for the FRONT
       (selfie) camera, where users expect to see themselves as in a
       mirror. The back camera should show the world exactly as the lens
       sees it. Both the video and the skeleton-overlay canvas must be
       toggled together, or the skeleton would visually misalign with
       the person's body whenever mirroring differs between the two. ---- */
    const shouldMirror = state.selectedCamera !== "back";
    const poseCanvasEl = $("poseCanvas");
    cameraVideo.classList.toggle("mirror", shouldMirror);
    if (poseCanvasEl) poseCanvasEl.classList.toggle("mirror", shouldMirror);

    $("cameraEmpty").classList.add("hidden");
    $("cameraStatus").textContent = "READY";
    $("liveFeedback").textContent = "Camera ready — press record.";

    startPoseTracking().catch(error => console.warn("Live pose tracking did not start:", error));
}


/* =========================================================
   SWITCH CAMERA
========================================================= */

async function switchCamera() {
    if (state.isRecording) {
        showToast("Stop recording before switching camera.");
        return;
    }

    state.selectedCamera = state.selectedCamera === "front" ? "back" : "front";

    $$("[data-camera]").forEach(button => {
        button.classList.toggle("active", button.dataset.camera === state.selectedCamera);
    });

    try {
        await startCamera();
        showToast(state.selectedCamera === "front" ? "Front camera selected." : "Back camera selected.");
    } catch (error) {
        console.error(error);
        showToast("Unable to switch camera.");
    }
}


/* =========================================================
   POSE ENGINE — LOAD / START / STOP
========================================================= */

async function ensurePoseLandmarker() {
    if (poseEngine.landmarker) return poseEngine.landmarker;
    if (poseEngine.failed) return null;

    if (poseEngine.loading) {
        while (poseEngine.loading) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return poseEngine.landmarker;
    }

    poseEngine.loading = true;
    setPoseStatus("loading", "◎ Skeleton: Loading...");
    console.log("[MovementCoach] Pose pipeline: starting MediaPipe load...");

    /* ---- NEW: watchdog — if this hangs (e.g. a CDN request that never
       resolves or rejects, rather than cleanly failing) for more than
       8s, surface it loudly instead of leaving the user staring at a
       silent "Loading..." chip forever with no explanation. ---- */
    const watchdogId = setTimeout(() => {
        if (!poseEngine.landmarker && !poseEngine.failed) {
            console.error("[MovementCoach] Pose pipeline: still not ready after 8s — likely a blocked/slow network request to cdn.jsdelivr.net or storage.googleapis.com.");
            showToast("Skeleton is taking unusually long to load. Check your internet connection or try a different network — MediaPipe loads its model from a CDN.");
            setPoseStatus("error", "Skeleton: taking too long — check connection");
        }
    }, 8000);

    try {
        if (!poseEngine.Vision) {
            console.log("[MovementCoach] Fetching @mediapipe/tasks-vision module...");
            poseEngine.Vision = await import(/* webpackIgnore: true */ POSE_VISION_MODULE_URL);
            console.log("[MovementCoach] tasks-vision module loaded.");
        }

        const { PoseLandmarker, FilesetResolver } = poseEngine.Vision;
        const filesetResolver = await FilesetResolver.forVisionTasks(POSE_WASM_BASE_URL);
        console.log("[MovementCoach] WASM fileset resolved. Creating landmarker...");

        let landmarker;
        try {
            landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
                baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
                runningMode: "VIDEO",
                numPoses: 1
            });
        } catch (gpuError) {
            console.warn("[MovementCoach] Pose GPU delegate failed, retrying on CPU:", gpuError);
            landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
                baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "CPU" },
                runningMode: "VIDEO",
                numPoses: 1
            });
        }

        clearTimeout(watchdogId);
        poseEngine.landmarker = landmarker;
        poseEngine.drawingUtils = null;
        setOnDeviceStatus("● Active");
        setPoseStatus("ready", "🦴 Skeleton: Ready");
        console.log("[MovementCoach] Pose pipeline: READY. Landmarker created successfully.");
        return landmarker;

    } catch (error) {
        clearTimeout(watchdogId);
        console.error("[MovementCoach] Pose pipeline FAILED:", error?.message || error, error);
        poseEngine.failed = true;
        setOnDeviceStatus("● Unavailable");
        setPoseStatus("error", "Skeleton unavailable — see console for exact error");
        /* ---- surface this failure visibly, not just in the console —
           the small status chip is easy to miss, and this is the #1
           reason skeleton/voice appear to "not work": the MediaPipe
           CDN or model file couldn't be reached. ---- */
        showToast("Skeleton tracking failed to load — check your internet connection (MediaPipe loads from a CDN) and reload.");
        return null;
    } finally {
        poseEngine.loading = false;
    }
}

function setPoseStatus(kind, text) {
    const chip = $("poseStatusChip");
    if (!chip) return;

    chip.textContent = text;
    chip.classList.remove("ready", "error");
    if (kind === "ready") chip.classList.add("ready");
    else if (kind === "error") chip.classList.add("error");
}

/* ---- FIX: setOnDeviceStatus was called in 3 places but never
   defined anywhere in the file — a ReferenceError that crashed
   ensurePoseLandmarker() immediately AFTER MediaPipe successfully
   loaded, right as it tried to report success. This is the actual
   root cause of the skeleton never appearing: the model loaded
   fine, then this crash aborted everything before the detection
   loop could ever start. This bug was already present in the
   original codebase before any of my changes. ---- */
function setOnDeviceStatus(text) {
    const dash = $("dashOnDeviceStatus");
    const live = $("onDeviceLiveStatus");
    if (dash) dash.textContent = text;
    if (live) live.textContent = text;
}

async function startPoseTracking() {
    if (!state.poseEnabled) {
        setPoseStatus("", "Skeleton: Off");
        return;
    }

    const video = $("cameraVideo");
    if (!video) return;

    const landmarker = await ensurePoseLandmarker();
    if (!landmarker) return;

    if (poseEngine.loopHandle) cancelAnimationFrame(poseEngine.loopHandle);

    poseEngine.lastFrameLandmarks = null;
    poseEngine.lastCueMessage = "";
    poseEngine.lastCueTime = -Infinity;
    poseEngine.cueCooldowns = {};
    poseEngine.angleBuffer = [];
    poseEngine.smoothedAngles = null;
    poseEngine.lastHudKey = "";
    poseEngine.noPoseFrames = 0;
    poseEngine.recognition = { pending: null, pendingHits: 0, locked: null, lockedConfidence: 0, lowHits: 0 };
    poseEngine.loopConfirmedRunning = false;
    poseEngine.firstDetectionLogged = false;
    console.log("[MovementCoach] startPoseTracking() called — camera stream present:", Boolean(state.cameraStream));
    setOnDeviceStatus("● Active");

    poseDetectionLoop();
}

function stopPoseTracking() {
    if (poseEngine.loopHandle) {
        cancelAnimationFrame(poseEngine.loopHandle);
        poseEngine.loopHandle = null;
    }
    clearPoseCanvas();
}

function clearPoseCanvas() {
    const canvas = $("poseCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
}


/* =========================================================
   POSE ENGINE — DETECTION LOOP
========================================================= */

function poseDetectionLoop() {
    poseEngine.loopHandle = requestAnimationFrame(poseDetectionLoop);

    if (!state.poseEnabled || !poseEngine.landmarker || !state.cameraStream) return;

    const video = $("cameraVideo");
    if (!video || video.readyState < 2 || !video.videoWidth) return;

    if (!poseEngine.loopConfirmedRunning) {
        poseEngine.loopConfirmedRunning = true;
        console.log("[MovementCoach] Pose detection loop is running — video ready, feeding frames to the model.");
    }

    const now = performance.now();
    if (now - poseEngine.lastDetectTime < POSE_DETECT_INTERVAL_MS) return;
    poseEngine.lastDetectTime = now;

    let result;
    try {
        result = poseEngine.landmarker.detectForVideo(video, now);
    } catch (error) {
        console.warn("Pose detection frame failed:", error);
        return;
    }

    const landmarks = result?.landmarks?.[0] || null;
    drawPoseOverlay(video, landmarks);

    if (!landmarks) {
        poseEngine.noPoseFrames++;
        updateRecognitionBanner(null, 0, "Please position yourself clearly in front of the camera.");
        showCameraQualityHint(poseEngine.noPoseFrames > 8 ? "No person detected. Step back so your full body is visible, and improve lighting." : null);
        return;
    }

    if (!poseEngine.firstDetectionLogged) {
        poseEngine.firstDetectionLogged = true;
        console.log("[MovementCoach] First body detected — skeleton should now be drawing on #poseCanvas.");
    }

    poseEngine.noPoseFrames = 0;

    /* ---- NEW: camera / pose quality guidance (Feature 13) ----
       Only warns when actual visibility conditions justify it —
       driven by MediaPipe's per-landmark visibility score, not a
       random timer. ---- */
    showCameraQualityHint(assessPoseQuality(landmarks));

    /* ---- NEW: compute joint angles for this frame ---- */
    const angles = computeFrameAngles(landmarks);

    pushAngleSample(angles, now);

    /* ---- NEW: automatic exercise recognition, with hysteresis ----
       classifyExercise() gives a raw per-tick guess; the code below
       is what actually stabilizes it using poseEngine.recognition
       and the RECOGNITION_* constants (both existed before but were
       unused — recognition was flipping on every noisy frame). ---- */
    if (state.autoDetectEnabled && modeData[state.selectedMode]?.trackable !== false) {

        if (now - poseEngine.lastClassifyTime > CLASSIFY_INTERVAL_MS) {
            poseEngine.lastClassifyTime = now;

            const { exercise: rawExercise, confidence: rawConfidence } = classifyExercise(poseEngine.angleBuffer);
            state.detectedExercise = rawExercise;
            state.detectionConfidence = rawConfidence;

            const { activeExercise, displayConfidence, guidance } = stabilizeRecognition(rawExercise, rawConfidence);

            state.activeExercise = activeExercise;

            if (activeExercise) {
                updateRecognitionBanner(activeExercise, displayConfidence, null);
            } else {
                updateRecognitionBanner(null, displayConfidence, guidance);
            }
        }

    } else {
        const manual = $("exerciseSelect")?.value || state.selectedMode;
        state.activeExercise = REP_CONFIG[manual] ? manual : null;
        hideRecognitionBanner();
    }

    /* ---- NEW: rep counting + form scoring while recording ---- */
    let repResult = null;
    if (state.isRecording) {

        repResult = updateRepCounting(state.activeExercise, angles, now);

        if (repResult && repResult.repCompleted) {
            state.sessionMetrics.reps = repResult.count;
            if (repResult.correct) {
                state.sessionMetrics.correctReps++;
            } else {
                state.sessionMetrics.corrections++;
            }
            flashHudRep();
        }
        const formScore = computeFormScore(state.activeExercise, angles);
        if (formScore != null) state.sessionMetrics.formScoreSamples.push(formScore);

        /* ---- NEW: this is what Feature 5 (multimodal Gemini analysis)
           actually needs — a compact per-frame snapshot of the joint
           angles and recognition confidence collected WHILE recording,
           so it can be summarized and sent to Gemini as supporting
           sensor evidence alongside the video, instead of Gemini only
           ever seeing raw pixels. ---- */
        state.sessionMetrics.jointSamples.push({
            knee: angles.kneeAvg,
            elbow: angles.elbowAvg,
            back: angles.backAngle
        });
        state.sessionMetrics.confidenceSamples.push(state.detectionConfidence || 0);

        updateFormHud(state.activeExercise, formScore, angles);
    }

    /* ---- existing: live spoken coaching cues ---- */
    if (state.coachEnabled) {
        const cue = evaluateLiveCues(landmarks, angles, now, state.activeExercise);
        if (cue) {
            // A correction always takes priority over praise this frame.
            maybeSpeakLiveCue(cue, now);
        } else if (repResult && repResult.repCompleted && repResult.correct) {
            // NEW: positive reinforcement — "Good rep." existed in the
            // translation dictionary but was never actually spoken
            // anywhere. Only fires when there's no active correction
            // needed this frame, and still respects the same cooldown
            // logic as corrections (won't spam every single good rep).
            maybeSpeakLiveCue("Good rep.", now);
        }
    }

    poseEngine.lastFrameLandmarks = landmarks;
    poseEngine.lastFrameTime = now;
}


function drawPoseOverlay(video, landmarks) {
    const canvas = $("poseCanvas");
    if (!canvas) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks) return;

    try {
        const { PoseLandmarker, DrawingUtils } = poseEngine.Vision;
        const drawingUtils = new DrawingUtils(ctx);

        /* ---- FIX: dot/line size was a fixed 3px value on a canvas
           sized to the camera's native resolution (e.g. 1280px wide).
           When the display shrinks that canvas — most notably in
           dance mode, where the camera panel is squeezed to half-width
           next to the reference video — a 3px dot becomes a sub-pixel
           speck and is effectively invisible. Sizes now scale with the
           canvas's actual pixel width so they stay clearly visible at
           any display size, plus a dark outline for contrast against
           any background/clothing color. ---- */
        const scale = canvas.width / 1280;
        const lineWidth = Math.max(4, 6 * scale);
        const dotRadius = Math.max(5, 8 * scale);

        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "#a78bfa", lineWidth });
        drawingUtils.drawLandmarks(landmarks, { radius: dotRadius, color: "#4ade80", fillColor: "#4ade80", lineWidth: 2 });
    } catch (error) {
        console.warn("Pose overlay draw failed:", error);
    }
}


/* =========================================================
   JOINT-ANGLE / GEOMETRY HELPERS
========================================================= */

function getPoint(landmarks, index) {
    const point = landmarks?.[index];
    if (!point) return null;
    if (typeof point.visibility === "number" && point.visibility < 0.4) return null;
    return { x: point.x, y: point.y };
}

function midpoint(a, b) {
    if (!a || !b) return null;
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function verticalLeanAngle(from, to) {
    if (!from || !to) return null;
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const radians = Math.atan2(Math.abs(dx), Math.abs(dy) || 0.0001);
    return radians * (180 / Math.PI);
}

function angleAt(a, b, c) {
    if (!a || !b || !c) return null;
    const v1 = { x: a.x - b.x, y: a.y - b.y };
    const v2 = { x: c.x - b.x, y: c.y - b.y };
    const dot = v1.x * v2.x + v1.y * v2.y;
    const mag1 = Math.hypot(v1.x, v1.y);
    const mag2 = Math.hypot(v2.x, v2.y);
    if (!mag1 || !mag2) return null;
    const cosine = Math.min(1, Math.max(-1, dot / (mag1 * mag2)));
    return Math.acos(cosine) * (180 / Math.PI);
}


/* =========================================================
   NEW: CAMERA / POSE QUALITY GUIDANCE (Feature 13)
   Reads MediaPipe's own per-landmark visibility scores —
   no guessing, no random timers. Only shown when the frame
   actually justifies it.
========================================================= */

const QUALITY_HINT_KEY_LANDMARKS = [
    POSE_LANDMARKS.LEFT_SHOULDER, POSE_LANDMARKS.RIGHT_SHOULDER,
    POSE_LANDMARKS.LEFT_HIP, POSE_LANDMARKS.RIGHT_HIP,
    POSE_LANDMARKS.LEFT_KNEE, POSE_LANDMARKS.RIGHT_KNEE,
    POSE_LANDMARKS.LEFT_ANKLE, POSE_LANDMARKS.RIGHT_ANKLE
];

function assessPoseQuality(landmarks) {
    if (!landmarks) return null;

    let visibleCount = 0;
    let lowVisCount = 0;
    let sumVisibility = 0;
    let sampled = 0;

    QUALITY_HINT_KEY_LANDMARKS.forEach(index => {
        const point = landmarks[index];
        if (!point) return;
        const visibility = typeof point.visibility === "number" ? point.visibility : 1;
        sampled++;
        sumVisibility += visibility;
        if (visibility >= 0.5) visibleCount++;
        if (visibility < 0.3) lowVisCount++;
    });

    if (!sampled) return null;

    const avgVisibility = sumVisibility / sampled;

    if (visibleCount < QUALITY_HINT_KEY_LANDMARKS.length - 2) {
        return "Make sure your full body is visible in the frame.";
    }

    if (avgVisibility < 0.45) {
        return "Improve lighting so the camera can track you clearly.";
    }

    const leftShoulder = landmarks[POSE_LANDMARKS.LEFT_SHOULDER];
    const rightShoulder = landmarks[POSE_LANDMARKS.RIGHT_SHOULDER];
    if (leftShoulder && rightShoulder) {
        const shoulderSpan = Math.abs(leftShoulder.x - rightShoulder.x);
        if (shoulderSpan < 0.09) return "Move farther from the camera so your whole body fits.";
        if (shoulderSpan > 0.55) return "Move closer to the camera.";
    }

    if (lowVisCount >= 3) return "Turn slightly toward the camera so more of your body is visible.";

    return null;
}

function showCameraQualityHint(message) {
    const hint = $("cameraQualityHint");
    if (!hint) return;

    if (!message) {
        hint.classList.add("hidden");
        return;
    }

    hint.textContent = "📷 " + message;
    hint.classList.remove("hidden");
}


/* =========================================================
   NEW: PER-FRAME JOINT ANGLE SNAPSHOT
   This is the shared feature set used by exercise
   recognition, rep counting and form scoring.
========================================================= */

function computeFrameAngles(landmarks) {
    const L = POSE_LANDMARKS;

    const nose = getPoint(landmarks, L.NOSE);
    const leftShoulder = getPoint(landmarks, L.LEFT_SHOULDER);
    const rightShoulder = getPoint(landmarks, L.RIGHT_SHOULDER);
    const leftElbow = getPoint(landmarks, L.LEFT_ELBOW);
    const rightElbow = getPoint(landmarks, L.RIGHT_ELBOW);
    const leftWrist = getPoint(landmarks, L.LEFT_WRIST);
    const rightWrist = getPoint(landmarks, L.RIGHT_WRIST);
    const leftHip = getPoint(landmarks, L.LEFT_HIP);
    const rightHip = getPoint(landmarks, L.RIGHT_HIP);
    const leftKnee = getPoint(landmarks, L.LEFT_KNEE);
    const rightKnee = getPoint(landmarks, L.RIGHT_KNEE);
    const leftAnkle = getPoint(landmarks, L.LEFT_ANKLE);
    const rightAnkle = getPoint(landmarks, L.RIGHT_ANKLE);

    const shoulderMid = midpoint(leftShoulder, rightShoulder);
    const hipMid = midpoint(leftHip, rightHip);
    const kneeMid = midpoint(leftKnee, rightKnee);
    const ankleMid = midpoint(leftAnkle, rightAnkle);
    const wristMid = midpoint(leftWrist, rightWrist);

    const leftElbowAngle = angleAt(leftShoulder, leftElbow, leftWrist);
    const rightElbowAngle = angleAt(rightShoulder, rightElbow, rightWrist);
    const leftKneeAngle = angleAt(leftHip, leftKnee, leftAnkle);
    const rightKneeAngle = angleAt(rightHip, rightKnee, rightAnkle);

    const elbowVals = [leftElbowAngle, rightElbowAngle].filter(v => v != null);
    const kneeVals = [leftKneeAngle, rightKneeAngle].filter(v => v != null);

    const elbowAvg = elbowVals.length ? elbowVals.reduce((a, b) => a + b, 0) / elbowVals.length : null;
    const kneeAvg = kneeVals.length ? kneeVals.reduce((a, b) => a + b, 0) / kneeVals.length : null;

    const backAngle = verticalLeanAngle(hipMid, shoulderMid);

    const ankleYDiff = (leftAnkle && rightAnkle) ? Math.abs(leftAnkle.y - rightAnkle.y) : null;

    const wristAboveHead =
        (leftWrist && nose && leftWrist.y < nose.y - 0.02) ||
        (rightWrist && nose && rightWrist.y < nose.y - 0.02);

    const horizontalBody = (shoulderMid && hipMid)
        ? Math.abs(shoulderMid.y - hipMid.y) < Math.abs(shoulderMid.x - hipMid.x)
        : false;

    const leftElbowNearTorso = (leftElbow && leftShoulder)
        ? Math.hypot(leftElbow.x - leftShoulder.x, leftElbow.y - leftShoulder.y) < 0.16
        : false;

    const rightElbowNearTorso = (rightElbow && rightShoulder)
        ? Math.hypot(rightElbow.x - rightShoulder.x, rightElbow.y - rightShoulder.y) < 0.16
        : false;

    const elbowNearTorso = leftElbowNearTorso || rightElbowNearTorso;

    const pressHeight = (shoulderMid && wristMid) ? (shoulderMid.y - wristMid.y) : null;

    return {
        nose, leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist,
        leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle,
        shoulderMid, hipMid, kneeMid, ankleMid, wristMid,
        leftElbowAngle, rightElbowAngle, leftKneeAngle, rightKneeAngle,
        elbowAvg, kneeAvg, backAngle, ankleYDiff, wristAboveHead, horizontalBody,
        elbowNearTorso, pressHeight
    };
}


/* =========================================================
   NEW: EXERCISE RECOGNITION (rule-based, on-device)
   Not a trained ML classifier — a transparent heuristic
   over joint-angle range-of-motion measured across the
   last ~1.6 seconds of frames. Good enough for a live demo,
   and safely reports low confidence instead of guessing.
========================================================= */

function pushAngleSample(angles, now) {
    poseEngine.angleBuffer.push({ ...angles, t: now });

    while (poseEngine.angleBuffer.length && now - poseEngine.angleBuffer[0].t > ANGLE_BUFFER_WINDOW_MS) {
        poseEngine.angleBuffer.shift();
    }
}

function rangeOf(values) {
    const clean = values.filter(v => Number.isFinite(v));
    if (clean.length < 2) return 0;
    return Math.max(...clean) - Math.min(...clean);
}

function fractionTrue(values) {
    if (!values.length) return 0;
    return values.filter(Boolean).length / values.length;
}

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function classifyExercise(buffer) {
    if (buffer.length < 6) return { exercise: null, confidence: 0 };

    const kneeROM = rangeOf(buffer.map(s => s.kneeAvg));
    const elbowROM = rangeOf(buffer.map(s => s.elbowAvg));
    const ankleYDiffAvg = buffer.map(s => s.ankleYDiff).filter(v => v != null)
        .reduce((sum, v, _, arr) => sum + v / arr.length, 0);
    const wristAboveHeadFrac = fractionTrue(buffer.map(s => s.wristAboveHead));
    const horizontalBodyFrac = fractionTrue(buffer.map(s => s.horizontalBody));
    const elbowNearTorsoFrac = fractionTrue(buffer.map(s => s.elbowNearTorso));

    const scores = {
        pushup: horizontalBodyFrac * 0.7 + clamp01(elbowROM / 90) * 0.3,
        shoulderpress: wristAboveHeadFrac * 0.6 + clamp01(elbowROM / 110) * 0.4,
        bicepcurl: Math.max(0, elbowNearTorsoFrac * 0.55 + clamp01(elbowROM / 110) * 0.45 - horizontalBodyFrac * 0.3 - wristAboveHeadFrac * 0.3),
        lunge: clamp01(ankleYDiffAvg / 0.12) * 0.55 + clamp01(kneeROM / 80) * 0.45,
        squat: clamp01(kneeROM / 80) * 0.7 + (1 - clamp01(ankleYDiffAvg / 0.12)) * 0.3
    };

    let best = null;
    let bestScore = -1;

    for (const [exercise, score] of Object.entries(scores)) {
        if (score > bestScore) {
            bestScore = score;
            best = exercise;
        }
    }

    return { exercise: best, confidence: Math.round(clamp01(bestScore) * 100) };
}

/* =========================================================
   NEW: RECOGNITION HYSTERESIS / STABILIZATION
   Turns the raw per-tick classifyExercise() guess into a
   stable "locked" exercise:
   - a brand-new exercise must win RECOGNITION_LOCK_HITS
     consecutive classify ticks before it becomes active
     (ignores a single noisy frame)
   - once locked, a competing exercise must beat the locked
     confidence by RECOGNITION_SWITCH_MARGIN AND also win
     RECOGNITION_LOCK_HITS consecutive ticks before switching
   - a transient confidence dip does not immediately drop the
     lock — it only clears after RECOGNITION_LOCK_HITS
     consecutive ticks below RECOGNITION_HOLD_CONFIDENCE
========================================================= */

function stabilizeRecognition(rawExercise, rawConfidence) {
    const rec = poseEngine.recognition;

    // Nothing locked yet — build confidence in a new candidate.
    if (!rec.locked) {
        if (rawExercise && rawConfidence >= CONFIDENCE_THRESHOLD) {
            if (rec.pending === rawExercise) {
                rec.pendingHits++;
            } else {
                rec.pending = rawExercise;
                rec.pendingHits = 1;
            }

            if (rec.pendingHits >= RECOGNITION_LOCK_HITS) {
                rec.locked = rawExercise;
                rec.lockedConfidence = rawConfidence;
                rec.pending = null;
                rec.pendingHits = 0;
                rec.lowHits = 0;
                return { activeExercise: rawExercise, displayConfidence: rawConfidence, guidance: null };
            }

            return { activeExercise: null, displayConfidence: rawConfidence, guidance: "Hold the position — confirming exercise." };
        }

        rec.pending = null;
        rec.pendingHits = 0;
        return { activeExercise: null, displayConfidence: rawConfidence, guidance: "Please position yourself clearly in front of the camera." };
    }

    // Already locked onto an exercise.
    if (rawExercise === rec.locked) {
        rec.lockedConfidence = rawConfidence;
        rec.lowHits = 0;
        rec.pending = null;
        rec.pendingHits = 0;
        return { activeExercise: rec.locked, displayConfidence: rawConfidence, guidance: null };
    }

    // A different exercise is winning this tick — only switch if it
    // clearly beats the current lock AND sustains across several ticks.
    if (rawExercise && rawConfidence >= rec.lockedConfidence + RECOGNITION_SWITCH_MARGIN) {
        if (rec.pending === rawExercise) {
            rec.pendingHits++;
        } else {
            rec.pending = rawExercise;
            rec.pendingHits = 1;
        }

        if (rec.pendingHits >= RECOGNITION_LOCK_HITS) {
            rec.locked = rawExercise;
            rec.lockedConfidence = rawConfidence;
            rec.pending = null;
            rec.pendingHits = 0;
            rec.lowHits = 0;
            return { activeExercise: rawExercise, displayConfidence: rawConfidence, guidance: null };
        }

        // Still mid-switch — keep coaching against the currently locked exercise.
        return { activeExercise: rec.locked, displayConfidence: rec.lockedConfidence, guidance: null };
    }

    rec.pending = null;
    rec.pendingHits = 0;

    // Confidence dipped but hasn't been low for long enough to drop the lock.
    if (rawConfidence < RECOGNITION_HOLD_CONFIDENCE) {
        rec.lowHits++;
        if (rec.lowHits >= RECOGNITION_LOCK_HITS) {
            rec.locked = null;
            rec.lockedConfidence = 0;
            rec.lowHits = 0;
            return { activeExercise: null, displayConfidence: rawConfidence, guidance: "Move into the exercise position for better recognition." };
        }
        return { activeExercise: rec.locked, displayConfidence: rec.lockedConfidence, guidance: null };
    }

    rec.lowHits = 0;
    return { activeExercise: rec.locked, displayConfidence: rec.lockedConfidence, guidance: null };
}

function updateRecognitionBanner(exercise, confidence, guidanceMessage) {
    const banner = $("recognitionBanner");
    const text = $("recognitionText");
    const fill = $("confidenceFill");
    if (!banner || !text || !fill) return;

    if (modeData[state.selectedMode]?.trackable === false) {
        banner.classList.add("hidden");
        return;
    }

    banner.classList.remove("hidden");

    if (guidanceMessage && !exercise) {
        banner.classList.add("low-confidence");
        text.textContent = guidanceMessage;
        fill.style.width = `${confidence}%`;
        return;
    }

    banner.classList.remove("low-confidence");
    text.textContent = `Exercise: ${EXERCISE_LABELS[exercise] || exercise} · Confidence: ${confidence}%`;
    fill.style.width = `${confidence}%`;
}

function hideRecognitionBanner() {
    $("recognitionBanner")?.classList.add("hidden");
}


/* updateRepCounting is now defined earlier, alongside repState,
   with cfg.minRange enforcement (see above). Old duplicate removed. */

function resetRepState() {
    repState.exercise = null;
    repState.phase = "rest";
    repState.count = 0;
    repState.correctCount = 0;
    repState.lastTransitionTime = 0;
    repState.extremum = null;
}


/* =========================================================
   NEW: FORM SCORING (0-100, continuous, per exercise)
========================================================= */

function computeFormScore(exercise, angles) {
    if (!exercise) return null;

    let score = 100;
    const backAngle = angles.backAngle;

    if (exercise === "squat" || exercise === "lunge") {
        if (backAngle != null && backAngle > 38) score -= Math.min(35, (backAngle - 38) * 1.5);
        if (angles.kneeMid && angles.ankleMid) {
            const offset = Math.abs(angles.kneeMid.x - angles.ankleMid.x);
            if (offset > 0.12) score -= Math.min(25, (offset - 0.12) * 200);
        }
    } else if (exercise === "pushup") {
        const bodyAngle = angleAt(angles.shoulderMid, angles.hipMid, angles.ankleMid);
        if (bodyAngle != null && bodyAngle < 155) score -= Math.min(35, (155 - bodyAngle) * 1.2);
    } else if (exercise === "bicepcurl") {
        if (!angles.elbowNearTorso) score -= 22;
    } else if (exercise === "shoulderpress") {
        if (backAngle != null && backAngle > 25) score -= Math.min(28, (backAngle - 25) * 1.5);
    } else {
        if (angles.leftShoulder && angles.rightShoulder) {
            const tilt = Math.abs(angles.leftShoulder.y - angles.rightShoulder.y);
            if (tilt > 0.05) score -= Math.min(20, (tilt - 0.05) * 300);
        }
    }

    return Math.max(0, Math.min(100, Math.round(score)));
}

function backAlignmentLabel(backAngle) {
    if (backAngle == null) return "--";
    if (backAngle <= 20) return "GOOD";
    if (backAngle <= 38) return "OK";
    return "FIX";
}


/* =========================================================
   NEW: LIVE FORM HUD
========================================================= */

function updateFormHud(exercise, formScore, angles) {
    const hud = $("formHud");
    if (!hud) return;

    if (!exercise || modeData[state.selectedMode]?.trackable === false) {
        hud.classList.add("hidden");
        return;
    }

    hud.classList.remove("hidden");

    const exerciseEl = $("hudExercise");
    const confidenceEl = $("hudConfidence");
    const scoreEl = $("hudFormScore");
    const repsEl = $("hudReps");
    const correctEl = $("hudCorrectReps");
    const kneeEl = $("hudKneeAngle");
    const backEl = $("hudBackAlignment");
    const primaryLabelEl = $("hudPrimaryLabel");

    /* ---- NEW: these two elements existed in the markup but were
       never populated — wire them to the live recognition state ---- */
    if (exerciseEl) exerciseEl.textContent = EXERCISE_LABELS[exercise] || exercise || "--";

    if (confidenceEl) {
        confidenceEl.textContent = state.autoDetectEnabled
            ? (state.detectionConfidence ? `${state.detectionConfidence}%` : "--")
            : "MANUAL";
    }

    if (scoreEl) {
        scoreEl.textContent = formScore != null ? `${formScore}` : "--";
        scoreEl.className = formScore == null ? "" : formScore >= 80 ? "good" : formScore >= 60 ? "warn" : "bad";
    }

    if (repsEl) repsEl.textContent = String(state.sessionMetrics.reps);
    if (correctEl) correctEl.textContent = String(state.sessionMetrics.correctReps);

    if (primaryLabelEl) primaryLabelEl.textContent = exercise === "bicepcurl" || exercise === "pushup" || exercise === "shoulderpress" ? "ELBOW" : "KNEE";

    if (kneeEl) {
        const primaryAngle = (exercise === "bicepcurl" || exercise === "pushup" || exercise === "shoulderpress") ? angles.elbowAvg : angles.kneeAvg;
        kneeEl.textContent = primaryAngle != null ? `${Math.round(primaryAngle)}°` : "--°";
    }

    if (backEl) {
        const label = backAlignmentLabel(angles.backAngle);
        backEl.textContent = label;
        backEl.className = label === "GOOD" ? "good" : label === "OK" ? "warn" : label === "FIX" ? "bad" : "";
    }
}

function flashHudRep() {
    const repsMetric = $("hudReps")?.closest(".hud-metric");
    if (!repsMetric) return;
    repsMetric.classList.remove("rep-flash");
    void repsMetric.offsetWidth;
    repsMetric.classList.add("rep-flash");
}


/* =========================================================
   LIVE SPOKEN COACHING CUES (extended for new exercises)
========================================================= */

function evaluateLiveCues(landmarks, angles, now, exercise) {

    /* ---- 1) movement speed check (applies to every mode) ---- */
    if (poseEngine.lastFrameLandmarks && poseEngine.lastFrameTime) {
        const dtSeconds = (now - poseEngine.lastFrameTime) / 1000;

        if (dtSeconds > 0.05) {
            const L = POSE_LANDMARKS;
            const prevWristL = getPoint(poseEngine.lastFrameLandmarks, L.LEFT_WRIST);
            const prevWristR = getPoint(poseEngine.lastFrameLandmarks, L.RIGHT_WRIST);
            const wristL = getPoint(landmarks, L.LEFT_WRIST);
            const wristR = getPoint(landmarks, L.RIGHT_WRIST);

            const speeds = [];
            if (prevWristL && wristL) speeds.push(Math.hypot(wristL.x - prevWristL.x, wristL.y - prevWristL.y) / dtSeconds);
            if (prevWristR && wristR) speeds.push(Math.hypot(wristR.x - prevWristR.x, wristR.y - prevWristR.y) / dtSeconds);

            const maxSpeed = speeds.length ? Math.max(...speeds) : 0;
            if (maxSpeed > 3.2) return "Slow down and control the movement.";
        }
    }

    if (!exercise) {
        /* ---- FIX: dance/yoga/wellness modes are marked trackable:false
           (no rep-counting exercise gets locked for them), which meant
           they fell into this branch and only ever got a narrow
           shoulder-tilt check — effectively silent for most of a
           session. General posture coaching now applies here too:
           back straightness, head/shoulder alignment, and movement
           speed (the speed check above already runs for every mode).
           This is what makes dance/yoga actually receive "keep your
           back straight" style corrections instead of near-silence. ---- */
        if (angles.backAngle != null && angles.backAngle > 34) return "Straighten your back.";

        /* ---- FIX: dance had zero arm/hand-position checking. Every
           existing hand/elbow cue lived inside exercise-specific
           branches (bicep curl, shoulder press) that never run during
           dance, since dance has no locked "exercise". This is a
           general-purpose check that works for freeform movement:
           if one hand is raised much higher than the other for a
           sustained moment, it's flagged — catching the "forgot to
           raise the other arm" / uneven arm mistake without needing
           a specific exercise to be recognized. ---- */
        if (angles.leftWrist && angles.rightWrist) {
            const handHeightDiff = Math.abs(angles.leftWrist.y - angles.rightWrist.y);
            if (handHeightDiff > 0.18) return "Keep both arms even.";
        }

        if (angles.leftShoulder && angles.rightShoulder) {
            const shoulderTilt = Math.abs(angles.leftShoulder.y - angles.rightShoulder.y);
            if (shoulderTilt > 0.07) return "Keep your shoulders aligned.";
        }

        if (angles.nose && angles.shoulderMid) {
            const headOffset = Math.abs(angles.nose.x - angles.shoulderMid.x);
            if (headOffset > 0.1) return "Keep your head aligned over your shoulders.";
        }

        return null;
    }

    if (exercise === "squat" || exercise === "lunge") {
        if (angles.backAngle != null && angles.backAngle > 38) return "Keep your back straighter.";
        if (angles.kneeMid && angles.ankleMid) {
            const kneeOffset = Math.abs(angles.kneeMid.x - angles.ankleMid.x);
            if (kneeOffset > 0.12) return "Keep your knee aligned with your foot.";
        }
        return null;
    }

    if (exercise === "pushup") {
        const bodyAngle = angleAt(angles.shoulderMid, angles.hipMid, angles.ankleMid);
        if (bodyAngle != null && bodyAngle < 155) return "Keep your back straight — engage your core.";
        return null;
    }

    if (exercise === "bicepcurl") {
        if (angles.leftElbow && angles.leftShoulder) {
            const leftOffset = Math.hypot(angles.leftElbow.x - angles.leftShoulder.x, angles.leftElbow.y - angles.leftShoulder.y);
            const rightOffset = (angles.rightElbow && angles.rightShoulder)
                ? Math.hypot(angles.rightElbow.x - angles.rightShoulder.x, angles.rightElbow.y - angles.rightShoulder.y)
                : 0;

            if (leftOffset > 0.18 && leftOffset > rightOffset) return "Keep your left elbow tucked in close to your body.";
            if (rightOffset > 0.18 && rightOffset > leftOffset) return "Keep your right elbow tucked in close to your body.";
        }
        return null;
    }

    if (exercise === "shoulderpress") {
        if (angles.leftWrist && angles.rightWrist) {
            const diff = angles.leftWrist.y - angles.rightWrist.y;
            if (Math.abs(diff) > 0.06) {
                return diff > 0
                    ? "Keep your left hand up — match your right arm's height."
                    : "Keep your right hand up — match your left arm's height.";
            }
        }
        if (angles.backAngle != null && angles.backAngle > 25) return "Avoid arching your back — keep your core engaged.";
        return null;
    }

    if (exercise === "standing") {
        if (angles.leftShoulder && angles.rightShoulder) {
            const shoulderTilt = Math.abs(angles.leftShoulder.y - angles.rightShoulder.y);
            if (shoulderTilt > 0.05) return "Keep your shoulders level.";
        }
        if (angles.nose && angles.shoulderMid) {
            const headOffset = Math.abs(angles.nose.x - angles.shoulderMid.x);
            if (headOffset > 0.09) return "Keep your head aligned over your shoulders.";
        }
        return null;
    }

    if (angles.leftShoulder && angles.rightShoulder) {
        const shoulderTilt = Math.abs(angles.leftShoulder.y - angles.rightShoulder.y);
        if (shoulderTilt > 0.07) return "Keep your shoulders aligned.";
    }

    return null;
}

function maybeSpeakLiveCue(message, now) {
    if (!("speechSynthesis" in window)) return;
    if (window.speechSynthesis.speaking) return;

    /* ---- FIX: previously a single flat cooldown applied to every
       cue regardless of content, so the identical correction (e.g.
       "keep your knees aligned") could repeat every ~4.5s for as
       long as the mistake persisted, which reads as nagging rather
       than natural coaching. A brand-new, different correction can
       still interrupt sooner; the SAME message now needs the longer
       POSE_SAME_CUE_GAP_MS gap before repeating. ---- */
    const isRepeat = message === poseEngine.lastCueMessage;
    const requiredGap = isRepeat ? POSE_SAME_CUE_GAP_MS : POSE_MIN_CUE_GAP_MS;
    if (now - poseEngine.lastCueTime < requiredGap) return;

    poseEngine.lastCueTime = now;
    poseEngine.lastCueMessage = message;

    /* ---- real-time cues are spoken AND shown in the selected
       voice language (English/Telugu/Hindi), not just tagged with
       the language code on untranslated English text. ---- */
    const localized = localizeCue(message);

    const liveFeedback = $("liveFeedback");
    if (liveFeedback) liveFeedback.textContent = "🔊 " + localized;

    showHudCue(localized);
    speakWithReferenceDucking(localized, state.voiceLanguage);
}

/* ---- NEW: surface the current spoken correction inside the Form HUD
   itself (element already existed in the markup but was never used),
   so judges can SEE the correction text next to the skeleton/reps,
   not just hear it. Fades out automatically. ---- */
let hudCueHideTimer = null;

function showHudCue(text) {
    const cueEl = $("hudCue");
    if (!cueEl) return;

    cueEl.textContent = "⚠️ " + text;
    cueEl.classList.remove("hidden");
    cueEl.style.opacity = "1";

    clearTimeout(hudCueHideTimer);
    hudCueHideTimer = setTimeout(() => {
        cueEl.style.opacity = "0";
        setTimeout(() => cueEl.classList.add("hidden"), 300);
    }, POSE_SAME_CUE_GAP_MS);
}

/* =========================================================
   NEW: AUDIO DUCKING FOR LIVE VOICE CORRECTIONS
   The reference video's background music plays at full volume
   (that's the intended dance-along experience). Rather than
   muting it or letting it fight the spoken correction for
   audibility, temporarily duck its volume down while the AI is
   actually speaking, then restore it the moment speech ends —
   the same technique radio/podcast apps use under voiceovers.
========================================================= */

const REFERENCE_DUCK_VOLUME = 0.18;
const REFERENCE_FULL_VOLUME = 1;

function speakWithReferenceDucking(text, language) {
    const reference = $("studioReference");
    const isNativeReferencePlaying = reference && !reference.paused && !reference.classList.contains("hidden");

    /* ---- NEW: duck the YouTube player's volume too, using the same
       IFrame API used for play/pause control. setVolume expects 0-100,
       unlike the native <video> element's 0-1 range. ---- */
    const ytPlayer = state.referenceSourceType === "youtube" ? state.studioYoutubePlayer : null;
    const isYoutubePlaying = ytPlayer && typeof ytPlayer.getPlayerState === "function" && ytPlayer.getPlayerState() === 1;

    if (isNativeReferencePlaying) {
        reference.volume = REFERENCE_DUCK_VOLUME;
    }
    if (isYoutubePlaying && typeof ytPlayer.setVolume === "function") {
        try { ytPlayer.setVolume(Math.round(REFERENCE_DUCK_VOLUME * 100)); } catch (error) { console.warn(error); }
    }

    speakText(text, language, () => {
        // Restore reference volume once this utterance finishes (or errors out),
        // but only if the user hasn't since stopped/hidden the reference video.
        if (reference && !reference.classList.contains("hidden")) {
            reference.volume = REFERENCE_FULL_VOLUME;
        }
        if (state.referenceSourceType === "youtube" && state.studioYoutubePlayer && typeof state.studioYoutubePlayer.setVolume === "function") {
            try { state.studioYoutubePlayer.setVolume(Math.round(REFERENCE_FULL_VOLUME * 100)); } catch (error) { console.warn(error); }
        }
    });
}


/* =========================================================
   REFERENCE STUDIO
========================================================= */

/* =========================================================
   NEW: YOUTUBE IFRAME PLAYER API (studio playback control)
   Only loaded when actually needed. Gives us play()/pause()/mute()/
   seekTo() control over an embedded YouTube reference during
   recording, mirroring what the native <video> element already does
   for file/url references — so beginRecording()/stopPractice() don't
   need separate code paths beyond a type check.
========================================================= */

let youtubeApiLoadPromise = null;

function loadYoutubeIframeApi() {
    if (window.YT && window.YT.Player) return Promise.resolve(window.YT);
    if (youtubeApiLoadPromise) return youtubeApiLoadPromise;

    youtubeApiLoadPromise = new Promise((resolve, reject) => {
        const previousCallback = window.onYouTubeIframeAPIReady;
        window.onYouTubeIframeAPIReady = () => {
            if (typeof previousCallback === "function") previousCallback();
            resolve(window.YT);
        };

        const script = document.createElement("script");
        script.src = "https://www.youtube.com/iframe_api";
        script.onerror = () => reject(new Error("Failed to load YouTube IFrame API"));
        document.head.appendChild(script);

        setTimeout(() => reject(new Error("YouTube API load timed out")), 10000);
    });

    return youtubeApiLoadPromise;
}

function destroyStudioYoutubePlayer() {
    if (state.studioYoutubePlayer && typeof state.studioYoutubePlayer.destroy === "function") {
        try { state.studioYoutubePlayer.destroy(); } catch (error) { console.warn(error); }
    }
    state.studioYoutubePlayer = null;
    const wrap = $("studioReferenceEmbedWrap");
    if (wrap) wrap.innerHTML = "";
}

async function setupStudioYoutubePlayer(videoId) {
    const wrap = $("studioReferenceEmbedWrap");
    if (!wrap) return;

    destroyStudioYoutubePlayer();
    wrap.innerHTML = "";
    const mount = document.createElement("div");
    mount.id = "studioYoutubeMount";
    wrap.appendChild(mount);

    try {
        const YT = await loadYoutubeIframeApi();
        state.studioYoutubePlayer = new YT.Player("studioYoutubeMount", {
            videoId,
            playerVars: { playsinline: 1, controls: 1, rel: 0 },
            events: {
                onReady: () => console.log("[MovementCoach] YouTube reference player ready."),
                onError: () => showToast("This YouTube video can't be embedded (the owner disabled embedding). Try a different link or upload a file instead."),
            },
        });
    } catch (error) {
        console.warn("YouTube player failed to load:", error);
        showToast("Couldn't load the YouTube player \u2014 check your connection.");
    }
}

function prepareReferenceForStudio() {
    const reference = $("studioReference");
    const embedWrap = $("studioReferenceEmbedWrap");
    const emptyEl = $("referenceEmpty");
    const DEFAULT_EMPTY_HTML = "<span>\uD83C\uDFAC</span><p>Upload a reference video</p>";

    // Always start clean — avoids stray players/state from a previous session,
    // and avoids the Instagram-specific message text leaking into later states.
    destroyStudioYoutubePlayer();
    embedWrap?.classList.add("hidden");
    reference.classList.add("hidden");
    if (emptyEl) emptyEl.innerHTML = DEFAULT_EMPTY_HTML;

    if (!state.referenceURL) {
        reference.removeAttribute("src");
        emptyEl?.classList.remove("hidden");
        $("referenceStatus").textContent = "NOT REQUIRED";
        return;
    }

    if (state.referenceSourceType === "youtube") {
        emptyEl?.classList.add("hidden");
        $("referenceStatus").textContent = "PREVIEW ONLY";
        embedWrap?.classList.remove("hidden");
        setupStudioYoutubePlayer(state.youtubeVideoId);
        return;
    }

    if (state.referenceSourceType === "instagram") {
        // No embeddable/controllable player available — be honest in the UI
        // rather than show a broken frame.
        if (emptyEl) emptyEl.innerHTML = "<span>\uD83D\uDCF7</span><p>Instagram link saved, but can't be played here \u2014 open it on Instagram to follow along.</p>";
        emptyEl?.classList.remove("hidden");
        $("referenceStatus").textContent = "LINK ONLY";
        return;
    }

    // "file" or "url" — identical native <video> handling either way.
    reference.src = state.referenceURL;
    reference.classList.remove("hidden");
    emptyEl?.classList.add("hidden");
    $("referenceStatus").textContent = "READY";
}


/* =========================================================
   STUDIO CONTROLS
========================================================= */

function setupStudioControls() {
    $("startPractice").addEventListener("click", startCountdown);
    $("stopPractice").addEventListener("click", () => stopPractice(true));
    $("studioCameraSwitch").addEventListener("click", switchCamera);

    $("studioPoseToggle")?.addEventListener("click", () => {
        state.poseEnabled = !state.poseEnabled;

        if (state.poseEnabled) {
            showToast("Live skeleton tracking on.");
            startPoseTracking().catch(error => console.warn(error));
        } else {
            showToast("Live skeleton tracking off.");
            stopPoseTracking();
        }
    });

    $("studioBack").addEventListener("click", () => {
        if (state.isRecording) stopPractice(false);
        else cleanupStudio();
        showPage(state.previousPage || "modeSetup");
    });
}


/* =========================================================
   COUNTDOWN
========================================================= */

function startCountdown() {
    if (state.isRecording || state.isCountingDown) return;

    if (!state.cameraStream) {
        showToast("Camera is not ready.");
        return;
    }

    state.isCountingDown = true;
    const overlay = $("countdownOverlay");
    const number = $("countdownNumber");

    overlay.classList.remove("hidden");
    let count = state.countdown;
    number.textContent = count;

    state.countdownTimer = setInterval(() => {
        count--;

        if (count <= 0) {
            clearInterval(state.countdownTimer);
            state.countdownTimer = null;
            overlay.classList.add("hidden");
            state.isCountingDown = false;
            beginRecording();
            return;
        }

        number.textContent = count;
    }, 1000);
}


/* =========================================================
   RECORDING
========================================================= */

function beginRecording() {
    if (!state.cameraStream) {
        showToast("Camera stream unavailable.");
        return;
    }

    state.recordedChunks = [];
    state.recordingBlob = null;

    if (state.recordingURL) {
        URL.revokeObjectURL(state.recordingURL);
        state.recordingURL = null;
    }

    /* ---- NEW: reset per-set rep/form metrics ---- */
    resetRepState();
    state.sessionMetrics = { reps: 0, correctReps: 0, corrections: 0, formScoreSamples: [], jointSamples: [], confidenceSamples: [] };

    const mimeTypes = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
    let mimeType = "";
    for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) { mimeType = type; break; }
    }

    try {
        state.mediaRecorder = mimeType
            ? new MediaRecorder(state.cameraStream, { mimeType })
            : new MediaRecorder(state.cameraStream);
    } catch (error) {
        console.error(error);
        showToast("Browser recording is not supported.");
        return;
    }

    state.mediaRecorder.ondataavailable = event => {
        if (event.data && event.data.size > 0) state.recordedChunks.push(event.data);
    };

    state.mediaRecorder.onstop = handleRecordingStopped;
    state.mediaRecorder.onerror = event => {
        console.error("Recorder error:", event.error);
        showToast("Recording error occurred.");
    };

    state.mediaRecorder.start(250);

    state.isRecording = true;
    state.practiceStartTime = Date.now();

    updateRecordingUI(true);
    startPracticeTimer();
    startReferencePlayback();

    $("liveFeedback").textContent = "Recording — movement captured.";
}


/* =========================================================
   REFERENCE PLAYBACK
========================================================= */

function startReferencePlayback() {
    /* ---- NEW: YouTube reference uses the IFrame Player API instead of
       the native <video> element's play()/currentTime/muted controls. ---- */
    if (state.referenceSourceType === "youtube") {
        const player = state.studioYoutubePlayer;
        if (player && typeof player.playVideo === "function") {
            try {
                player.seekTo(0, true);
                player.unMute();
                player.playVideo();
            } catch (error) {
                console.warn("YouTube playback start failed:", error);
            }
        }
        return;
    }
    if (state.referenceSourceType === "instagram") return; // no controllable player

    const reference = $("studioReference");
    if (!reference || reference.classList.contains("hidden") || !state.referenceURL) return;

    reference.currentTime = 0;

    /* ---- Reference video's own background music plays normally —
       it's the intended reference experience (e.g. dance music to
       move along to). It is NOT muted. What IS kept separate:
       - cameraVideo (your own camera preview) stays muted, so your
         mic audio is captured into the recording but never played
         back out loud through the speakers during the take — this
         is what avoids feedback-loop / "multiple audio streams
         fighting". See speakWithReferenceDucking() below for how
         AI voice corrections stay audible over the reference music
         instead of the two colliding. ---- */
    reference.muted = false;
    reference.volume = 1;

    const playPromise = reference.play();
    if (playPromise) {
        playPromise.catch(error => {
            console.warn("Reference autoplay prevented:", error);
            showToast("Press play on the reference video if browser blocks autoplay.");
        });
    }
}


/* =========================================================
   STOP
========================================================= */

function stopPractice(showResults = true) {
    if (state.isCountingDown) {
        cancelCountdown();
        showToast("Countdown cancelled.");
        return;
    }

    if (!state.isRecording) {
        if (showResults) showToast("No active recording.");
        return;
    }

    state.isRecording = false;
    updateRecordingUI(false);
    stopPracticeTimer();
    stopReferenceVideo();

    if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
        state.mediaRecorder.stop();
    } else {
        handleRecordingStopped();
    }

    $("liveFeedback").textContent = "Recording stopped.";
    $("cameraStatus").textContent = "PROCESSING";

    if (showResults) $("studioHint").textContent = "Uploading video to AI...";
}


/* =========================================================
   RECORDING FINISHED
========================================================= */

function handleRecordingStopped() {
    if (state.recordedChunks.length === 0) {
        showToast("No recording data was captured.");
        cleanupStudio();
        return;
    }

    state.recordingBlob = new Blob(state.recordedChunks, { type: "video/webm" });
    state.recordingURL = URL.createObjectURL(state.recordingBlob);

    stopCameraOnly();

    const reviewVideo = $("reviewVideo");
    reviewVideo.src = state.recordingURL;
    reviewVideo.load();

    const duration = Math.max(1, (Date.now() - (state.practiceStartTime || Date.now())) / 1000);

    const exerciseUsed = state.activeExercise || $("exerciseSelect")?.value || state.selectedMode;

    const avgFormScore = state.sessionMetrics.formScoreSamples.length
        ? Math.round(state.sessionMetrics.formScoreSamples.reduce((a, b) => a + b, 0) / state.sessionMetrics.formScoreSamples.length)
        : null;

    /* ---- NEW: summarize joint-angle and confidence samples collected
       during recording into small averages Gemini can use as
       supporting evidence (Feature 5) ---- */
    const jointMetrics = summarizeJointSamples(state.sessionMetrics.jointSamples);
    const avgConfidence = state.sessionMetrics.confidenceSamples.length
        ? Math.round(state.sessionMetrics.confidenceSamples.reduce((a, b) => a + b, 0) / state.sessionMetrics.confidenceSamples.length)
        : null;

    state.currentSession = {
        id: Date.now(),
        mode: state.selectedMode,
        modeTitle: modeData[state.selectedMode].title,
        exercise: exerciseUsed,
        duration,
        recordingURL: state.recordingURL,
        recordingBlob: state.recordingBlob,
        createdAt: new Date().toISOString(),
        score: null,
        breakdown: null,
        mistakes: [],
        translations: null,
        originalFeedback: "",

        /* ---- NEW: session metrics captured live during the set ---- */
        metrics: {
            reps: state.sessionMetrics.reps,
            correctReps: state.sessionMetrics.correctReps,
            corrections: state.sessionMetrics.corrections,
            avgFormScore,
            avgConfidence,
            jointMetrics
        }
    };

    showPage("analysis");
    populateWorkoutSummary(state.currentSession);
    runAnalysis();
}


/* ---- NEW: turns the raw per-frame joint sample array collected
   during recording into a small set of averages — this is the
   "jointMetrics" payload sent to Gemini as supporting sensor
   evidence (Feature 5), not the full per-frame stream. ---- */
function summarizeJointSamples(samples) {
    if (!samples || !samples.length) return null;

    const avg = key => {
        const vals = samples.map(s => s[key]).filter(v => Number.isFinite(v));
        if (!vals.length) return null;
        return Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10;
    };

    return {
        avgKneeAngle: avg("knee"),
        avgElbowAngle: avg("elbow"),
        avgBackLeanAngle: avg("back"),
        sampleCount: samples.length
    };
}


/* =========================================================
   NEW: WORKOUT SUMMARY (reps / accuracy / corrections / duration)
========================================================= */

function populateWorkoutSummary(session) {
    const metrics = session.metrics || {};

    $("summaryExercise").textContent = EXERCISE_LABELS[session.exercise] || session.exercise || "--";
    $("summaryReps").textContent = metrics.reps || metrics.reps === 0 ? String(metrics.reps) : "--";
    $("summaryCorrectReps").textContent = metrics.correctReps || metrics.correctReps === 0 ? String(metrics.correctReps) : "--";
    $("summaryCorrections").textContent = metrics.corrections || metrics.corrections === 0 ? String(metrics.corrections) : "--";
    $("summaryDuration").textContent = formatTime(session.duration);

    /* ---- NEW: Live AI score badge is available immediately from
       on-device metrics, before Gemini responds ---- */
    const liveBadge = $("liveScoreBadge");
    if (liveBadge) liveBadge.textContent = metrics.avgFormScore != null ? `${metrics.avgFormScore}%` : "--";
}

/* =========================================================
   NEW: SESSION INTELLIGENCE SUMMARY (Feature 7)
   Fills in the strongest / weakest / major-issue / Gemini-score
   cards that existed in the markup but were never populated —
   derived only from real breakdown + mistake data, no invented
   numbers.
========================================================= */

function populateSessionIntelligence(session, result) {
    const geminiBadge = $("geminiScoreBadge");
    if (geminiBadge) geminiBadge.textContent = Number.isFinite(result.score) ? `${result.score}%` : "--";

    const breakdown = result.breakdown || {};
    const labels = { left: "Left Side", right: "Right Side", upper: "Upper Body", lower: "Lower Body" };
    const entries = Object.entries(breakdown).filter(([, value]) => Number.isFinite(value));

    if (entries.length) {
        entries.sort((a, b) => b[1] - a[1]);
        const strongestKey = entries[0][0];
        const weakestKey = entries[entries.length - 1][0];

        $("summaryStrongest").textContent = `${labels[strongestKey] || strongestKey} (${entries[0][1]}%)`;
        $("summaryWeakest").textContent = `${labels[weakestKey] || weakestKey} (${entries[entries.length - 1][1]}%)`;
    } else {
        $("summaryStrongest").textContent = "--";
        $("summaryWeakest").textContent = "--";
    }

    const mistakes = result.mistakes || [];
    $("summaryMajorMistake").textContent = mistakes.length ? mistakes[0].title : "None detected";
}


/* =========================================================
   CLEANUP
========================================================= */

function stopReferenceVideo() {
    if (state.referenceSourceType === "youtube") {
        const player = state.studioYoutubePlayer;
        if (player && typeof player.pauseVideo === "function") {
            try { player.pauseVideo(); } catch (error) { console.warn(error); }
        }
        return;
    }
    if (state.referenceSourceType === "instagram") return;

    const reference = $("studioReference");
    if (!reference) return;
    reference.pause();
    try { reference.currentTime = 0; } catch (error) { console.warn(error); }
    reference.muted = true;
}

function stopCameraOnly() {
    stopPoseTracking();

    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => {
            try { track.stop(); } catch (error) { console.warn(error); }
        });
    }

    state.cameraStream = null;

    const cameraVideo = $("cameraVideo");
    if (cameraVideo) {
        cameraVideo.pause();
        cameraVideo.srcObject = null;
    }
}

function cleanupStudio() {
    cancelCountdown();
    stopPracticeTimer();
    stopReferenceVideo();
    stopCameraOnly();
    destroyStudioYoutubePlayer();

    if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
        try { state.mediaRecorder.stop(); } catch (error) { console.warn(error); }
    }

    state.isRecording = false;
    updateRecordingUI(false);
}


/* =========================================================
   COUNTDOWN CANCEL
========================================================= */

function cancelCountdown() {
    if (state.countdownTimer) {
        clearInterval(state.countdownTimer);
        state.countdownTimer = null;
    }
    state.isCountingDown = false;
    $("countdownOverlay").classList.add("hidden");
}


/* =========================================================
   RECORDING UI
========================================================= */

function updateRecordingUI(recording) {
    const button = $("startPractice");
    const indicator = $("recordingIndicator");

    if (recording) {
        button.classList.add("recording");
        indicator.classList.remove("hidden");
        $("referenceStatus").textContent = "PLAYING";
        $("cameraStatus").textContent = "RECORDING";
        $("liveScore").textContent = "LIVE";
        $("studioHint").textContent = "Recording in progress. Press ■ to stop.";
    } else {
        button.classList.remove("recording");
        indicator.classList.add("hidden");
        $("liveScore").textContent = "READY";
        $("studioHint").textContent = "Recording stopped.";
    }
}


/* =========================================================
   TIMER
========================================================= */

function startPracticeTimer() {
    stopPracticeTimer();
    $("practiceTimer").textContent = "00:00";

    state.timerInterval = setInterval(() => {
        if (!state.practiceStartTime) return;
        const seconds = Math.floor((Date.now() - state.practiceStartTime) / 1000);
        $("practiceTimer").textContent = formatTime(seconds);
    }, 250);
}

function stopPracticeTimer() {
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }
}


/* =========================================================
   RESET STUDIO
========================================================= */

function resetStudioUI() {
    $("practiceTimer").textContent = "00:00";
    $("liveScore").textContent = "READY";
    $("cameraStatus").textContent = "READY";
    $("referenceStatus").textContent = state.referenceURL ? "READY" : "NOT REQUIRED";
    $("recordingIndicator").classList.add("hidden");
    $("liveFeedback").textContent = "Camera ready.";
    $("studioHint").textContent = "Press record when you are ready.";

    setPoseStatus(state.poseEnabled ? "loading" : "", state.poseEnabled ? "◎ Skeleton: Loading..." : "Skeleton: Off");

    resetRepState();
    state.sessionMetrics = { reps: 0, correctReps: 0, corrections: 0, formScoreSamples: [], jointSamples: [], confidenceSamples: [] };
    state.detectedExercise = null;
    state.detectionConfidence = 0;
    state.activeExercise = null;

    hideRecognitionBanner();
    $("formHud")?.classList.add("hidden");
    $("cameraQualityHint")?.classList.add("hidden");
    $("hudCue")?.classList.add("hidden");
    clearTimeout(hudCueHideTimer);
    poseEngine.noPoseFrames = 0;

    prepareReferenceForStudio();
}


/* =========================================================
   REAL AI ANALYSIS (Gemini backend, unchanged endpoint)
========================================================= */

async function runAnalysis() {
    if (!state.currentSession) return;

    $("overallScore").textContent = "--";
    $("resultTitle").textContent = "Analyzing movement...";
    $("resultText").textContent = "Gemini is comparing your movement with the reference.";
    $("mistakeList").innerHTML = `<div class="empty-message">🤖 AI is analyzing body position, timing and movement consistency...</div>`;
    $("coachRecommendationCard").innerHTML = `<div class="empty-message">Generating your personalized recommendation...</div>`;
    $("geminiScoreBadge").textContent = "--";
    $("summaryStrongest").textContent = "--";
    $("summaryWeakest").textContent = "--";
    $("summaryMajorMistake").textContent = "--";
    setGeminiStatusPills("● Analyzing session...");

    try {
        const formData = new FormData();
        formData.append("video", state.currentSession.recordingBlob, "practice.webm");

        if (state.referenceFile) {
            formData.append("reference", state.referenceFile, state.referenceFile.name);
        } else if (state.referenceSourceType === "url" && state.referenceURL) {
            /* ---- NEW: a pasted direct-video-file link has no local File
               object to upload — the backend fetches it server-side
               instead (no CORS restriction there) and treats it exactly
               like an uploaded reference file. YouTube/Instagram are
               intentionally NOT sent here; see loadReferenceFromUrl(). ---- */
            formData.append("referenceUrl", state.referenceURL);
        }

        formData.append("mode", state.selectedMode);
        formData.append("exercise", state.currentSession.exercise || $("exerciseSelect").value);
        formData.append("language", state.voiceLanguage);

        /* ---- NEW: Feature 5 — send the on-device session metrics
           collected by MediaPipe alongside the video, so Gemini
           analyzes "Computer Vision + Sensor Data" together instead
           of the video alone. These were computed live and are sent
           as-is; the backend prompt explicitly tells Gemini to treat
           them as supporting evidence, not ground truth, and to keep
           relying on what it actually sees in the video. ---- */
        const metrics = state.currentSession.metrics || {};
        formData.append("reps", String(metrics.reps ?? 0));
        formData.append("correctReps", String(metrics.correctReps ?? 0));
        formData.append("corrections", String(metrics.corrections ?? 0));
        formData.append("liveFormScore", metrics.avgFormScore != null ? String(metrics.avgFormScore) : "");
        formData.append("recognitionConfidence", metrics.avgConfidence != null ? String(metrics.avgConfidence) : "");
        formData.append("duration", String(Math.round(state.currentSession.duration || 0)));
        formData.append("jointMetrics", metrics.jointMetrics ? JSON.stringify(metrics.jointMetrics) : "");

        const response = await fetch("/api/analyze", { method: "POST", body: formData });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Backend ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        const normalized = normalizeAnalysisResult(data);
        applyAnalysisResult(normalized);
        setGeminiStatusPills("● Ready");
        showToast("AI analysis complete.");

    } catch (error) {
        console.error("AI analysis failed:", error);

        $("resultTitle").textContent = "Analysis failed";
        $("resultText").textContent = "The AI server could not complete this analysis. Check the backend terminal.";
        $("mistakeList").innerHTML = `<div class="empty-message">AI analysis unavailable. Please check that server.js is running and GEMINI_API_KEY is configured.</div>`;
        $("coachRecommendationCard").innerHTML = `<div class="empty-message">Recommendation unavailable — the AI server did not respond.</div>`;
        setGeminiStatusPills("● Offline");

        showToast("AI analysis failed. Check server.");
    }
}


/* =========================================================
   NORMALIZE RESULT
========================================================= */

function normalizeAnalysisResult(data) {
    const score = clampScore(Number(data.score ?? data.overallScore ?? data.overall ?? 0));
    const breakdownSource = data.breakdown || {};

    const breakdown = {
        left: clampScore(Number(breakdownSource.left ?? score)),
        right: clampScore(Number(breakdownSource.right ?? score)),
        upper: clampScore(Number(breakdownSource.upper ?? score)),
        lower: clampScore(Number(breakdownSource.lower ?? score))
    };

    const rawMistakes = Array.isArray(data.mistakes) ? data.mistakes : [];

    const mistakes = rawMistakes.map((mistake, index) => {
        const time = Number(mistake.time ?? mistake.seconds ?? mistake.timestampSeconds ?? 0);
        return {
            id: mistake.id ?? index + 1,
            time: Math.max(0, time),
            timestamp: mistake.timestamp ?? formatTime(time),
            title: mistake.title ?? mistake.type ?? mistake.issue ?? `Movement issue ${index + 1}`,
            description: mistake.description ?? mistake.feedback ?? mistake.message ?? "Movement needs improvement.",
            correction: mistake.correction ?? mistake.fix ?? mistake.howToFix ?? ""
        };
    }).sort((a, b) => a.time - b.time);

    return {
        score,
        breakdown,
        mistakes,
        title: data.title ?? (score >= 90 ? "Excellent movement" : score >= 75 ? "Good progress" : "Keep practicing"),
        text: data.feedback ?? data.message ?? data.summary ?? generateResultText(score, mistakes.length),
        translations: data.translations || null
    };
}


/* =========================================================
   APPLY RESULT
========================================================= */

function applyAnalysisResult(result) {
    if (!state.currentSession) return;

    state.currentSession.score = result.score;
    state.currentSession.breakdown = result.breakdown;
    state.currentSession.mistakes = result.mistakes;
    state.currentSession.translations = result.translations || null;
    state.currentSession.originalFeedback = result.text || "";

    state.currentMistakes = result.mistakes || [];
    state.currentTranslations = result.translations || null;
    state.currentFeedbackText = result.text || "";

    $("overallScore").textContent = result.score;
    $("resultTitle").textContent = result.title;
    $("resultText").textContent = result.text;

    setBreakdown("left", result.breakdown.left);
    setBreakdown("right", result.breakdown.right);
    setBreakdown("upper", result.breakdown.upper);
    setBreakdown("lower", result.breakdown.lower);

    renderMistakes(result.mistakes);
    updateScoreRing(result.score);
    populateSessionIntelligence(state.currentSession, result);

    const feedbackSelect = $("feedbackLanguage");
    if (feedbackSelect) {
        feedbackSelect.value = state.voiceLanguage;
        state.feedbackLanguage = state.voiceLanguage;
    }

    /* ---- FIX: setting a <select>'s .value in JS does NOT fire its
       'change' event, so translateCurrentFeedback() — which is only
       wired to the dropdown's change listener — was never actually
       running automatically. This meant the results screen always
       displayed Gemini's default English text/mistakes first, even
       when Telugu or Hindi was selected the whole session, until the
       user manually re-touched the dropdown. Explicitly apply the
       selected language now so the very first paint is already
       correct. ---- */
    translateCurrentFeedback(state.voiceLanguage);

    saveSession();
    updateProgressUI();
    checkAndUnlockAchievements(state.currentSession);
    fetchCoachRecommendation(state.currentSession);

    if (state.coachEnabled) speakShortCoachSummary(result);
}


/* =========================================================
   BREAKDOWN
========================================================= */

function setBreakdown(name, value) {
    const scoreElement = $(`${name}Score`);
    const progressElement = $(`${name}Progress`);
    if (!scoreElement || !progressElement) return;

    scoreElement.textContent = `${value}%`;
    progressElement.style.width = `${value}%`;
}


/* =========================================================
   SCORE RING
========================================================= */

function updateScoreRing(score) {
    const circle = $("scoreRing");
    const circumference = 326.7;
    const offset = circumference - (circumference * score) / 100;
    circle.style.strokeDashoffset = offset;
}


/* =========================================================
   MISTAKES
========================================================= */

function renderMistakes(mistakes) {
    const container = $("mistakeList");
    $("mistakeCount").textContent = `${mistakes.length} ${mistakes.length === 1 ? "issue" : "issues"}`;

    if (!mistakes.length) {
        container.innerHTML = `<div class="empty-message">✅ No major movement mistakes detected.</div>`;
        updateMistakeBadge();
        return;
    }

    container.innerHTML = mistakes.map(mistake => `
        <div class="mistake-card" data-mistake-time="${mistake.time}">
            <div class="mistake-time">${escapeHTML(mistake.timestamp || formatTime(mistake.time))}</div>
            <div class="mistake-icon">!</div>
            <div class="mistake-content">
                <strong>${escapeHTML(mistake.title)}</strong>
                <span>${escapeHTML(mistake.description)}</span>
                ${mistake.correction ? `<span class="mistake-fix"><b>${escapeHTML(fixLabelForLanguage())}:</b> ${escapeHTML(mistake.correction)}</span>` : ""}
            </div>
            <button class="mistake-speak-btn" data-speak-mistake="${escapeHTML(mistake.id)}" title="Speak this feedback">🔊</button>
        </div>
    `).join("");

    container.querySelectorAll(".mistake-card").forEach(card => {
        card.addEventListener("click", event => {
            if (event.target.closest(".mistake-speak-btn")) return;
            seekReviewVideo(Number(card.dataset.mistakeTime));
        });
    });

    updateMistakeBadge();
}

function fixLabelForLanguage() {
    const lang = (state.feedbackLanguage || state.voiceLanguage || "en-IN");
    if (lang.startsWith("te")) return "ఎలా సరిదిద్దాలి";
    if (lang.startsWith("hi")) return "कैसे सुधारें";
    return "How to fix";
}


/* =========================================================
   SEEK VIDEO
========================================================= */

function seekReviewVideo(time) {
    const video = $("reviewVideo");
    if (!video.src) {
        showToast("Recording is not available.");
        return;
    }

    video.currentTime = Math.max(0, time);
    video.play().catch(() => {});
    video.scrollIntoView({ behavior: "smooth", block: "center" });
}


/* =========================================================
   VOICE SETUP
========================================================= */

function setupFeedbackVoice() {
    const languageSelect = $("feedbackLanguage");
    const speakButton = $("speakFeedback");
    const coachButton = $("playVoiceCoach");

    if (languageSelect) {
        languageSelect.addEventListener("change", async () => {
            state.feedbackLanguage = languageSelect.value;
            setVoiceLanguage(languageSelect.value, false);
            await translateCurrentFeedback(state.feedbackLanguage);
            showToast(`Feedback language: ${getLanguageName(state.feedbackLanguage)}`);
        });
    }

    if (speakButton) {
        speakButton.addEventListener("click", () => {
            if (window.speechSynthesis.speaking) {
                window.speechSynthesis.cancel();
                speakButton.textContent = "🔊 Speak Feedback";
                return;
            }
            speakCurrentFeedback();
        });
    }

    if (coachButton) coachButton.addEventListener("click", playVoiceCoaching);

    document.addEventListener("click", event => {
        const button = event.target.closest("[data-speak-mistake]");
        if (!button) return;

        const mistakeId = button.dataset.speakMistake;
        const mistake = state.currentMistakes.find(item => String(item.id) === String(mistakeId));
        if (mistake) speakMistake(mistake);
    });
}


/* =========================================================
   TRANSLATION
========================================================= */

async function translateCurrentFeedback(language) {
    const session = state.currentSession;
    if (!session) return;

    const languageKey = language.startsWith("te") ? "te" : language.startsWith("hi") ? "hi" : "en";
    const translations = session.translations;

    if (translations && translations[languageKey]) {
        const translated = translations[languageKey];

        if (translated.summary) {
            $("resultText").textContent = translated.summary;
            state.currentFeedbackText = translated.summary;
        }

        if (Array.isArray(translated.mistakes)) {
            const translatedMistakes = translated.mistakes.map((item, index) => {
                const original = session.mistakes[index] || {};
                return {
                    id: original.id ?? index + 1,
                    time: original.time ?? 0,
                    timestamp: original.timestamp ?? formatTime(original.time || 0),
                    title: item.title || original.title || "Movement issue",
                    description: item.description || original.description || "",
                    correction: item.correction || original.correction || ""
                };
            });

            state.currentMistakes = translatedMistakes;
            renderMistakes(translatedMistakes);
        }

        return;
    }

    if (languageKey === "en") {
        state.currentFeedbackText = session.originalFeedback || "";
        $("resultText").textContent = state.currentFeedbackText;
        state.currentMistakes = session.mistakes || [];
        renderMistakes(session.mistakes || []);
        return;
    }

    showToast("This language was not returned by the AI.");
}


/* =========================================================
   SPEECH
========================================================= */

function speakCurrentFeedback() {
    if (!state.currentSession) {
        showToast("No AI feedback available.");
        return;
    }

    let text = state.currentFeedbackText;
    if (!text) text = buildSpeechFeedback(state.currentSession);
    speakText(text, state.feedbackLanguage);
}

function speakMistake(mistake) {
    if (!mistake) return;
    const parts = [mistake.title, mistake.description];
    if (mistake.correction) parts.push(mistake.correction);
    speakText(parts.filter(Boolean).join(". ") + ".", state.feedbackLanguage);
}

/* =========================================================
   NEW: VOICE AVAILABILITY DIAGNOSTIC
   Many phones/browsers don't ship a Telugu or Hindi speech
   voice by default. If that's the case, speechSynthesis will
   silently fall back to a default voice (often reading nothing,
   or reading it in the wrong accent) instead of throwing an
   error — so this is checked once and surfaced clearly instead
   of leaving it looking like a silent bug.
========================================================= */
let voiceAvailabilityWarned = {};

function checkVoiceAvailability(language) {
    if (!("speechSynthesis" in window)) return;
    if (voiceAvailabilityWarned[language]) return;

    const langPrefix = (language || "en-IN").split("-")[0];
    const voices = window.speechSynthesis.getVoices();

    // Voice list loads asynchronously in some browsers — if empty, wait once and re-check.
    if (!voices.length) {
        window.speechSynthesis.onvoiceschanged = () => checkVoiceAvailability(language);
        return;
    }

    const hasMatch = voices.some(v => v.lang && v.lang.toLowerCase().startsWith(langPrefix));
    voiceAvailabilityWarned[language] = true;

    if (!hasMatch && langPrefix !== "en") {
        const langName = getLanguageName(language);
        showToast(`This browser/device has no ${langName} voice installed — spoken feedback will fall back to the device's default voice. Text feedback (HUD + mistake cards) still shows in ${langName}.`);
    }
}

function speakText(text, language, onDone) {
    if (!("speechSynthesis" in window)) {
        showToast("Voice feedback is not supported by this browser.");
        return;
    }
    checkVoiceAvailability(language);
    if (!text) {
        showToast("No feedback available to speak.");
        return;
    }

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = language || "en-IN";
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.volume = 1;

    const button = $("speakFeedback");

    utterance.onstart = () => { if (button) button.textContent = "⏹ Stop Speaking"; };
    utterance.onend = () => {
        if (button) button.textContent = "🔊 Speak Feedback";
        if (onDone) onDone();
    };
    utterance.onerror = error => {
        console.warn("Speech error:", error);
        if (button) button.textContent = "🔊 Speak Feedback";
        if (onDone) onDone();
    };

    window.speechSynthesis.speak(utterance);
}

function buildSpeechFeedback(session) {
    const score = session.score || 0;
    const mistakes = session.mistakes || [];
    const lang = state.feedbackLanguage || state.voiceLanguage || "en-IN";

    let text = phrase("scoreIs", lang, score) + " ";

    if (!mistakes.length) {
        return text + phrase("excellentNoMistakes", lang);
    }

    text += phrase("foundAreas", lang, mistakes.length) + " ";

    mistakes.slice(0, 5).forEach((mistake, index) => {
        text += `${phrase("issueN", lang, index + 1)} ${mistake.title}. ${mistake.description}. `;
        if (mistake.correction) text += `${phrase("toFixIt", lang)} ${mistake.correction}. `;
    });

    return text;
}

function getLanguageName(language) {
    const languages = { "en-IN": "English", "te-IN": "Telugu", "hi-IN": "Hindi" };
    return languages[language] || "English";
}


/* =========================================================
   VOICE AI COACH (queued, paced speech)
========================================================= */

function speakShortCoachSummary(result) {
    if (!("speechSynthesis" in window)) return;

    const lang = state.voiceLanguage || "en-IN";
    const score = result.score;
    // Pull the top mistake from the already-localized list (translateCurrentFeedback
    // runs before this is called), not the raw English result.mistakes.
    const topMistake = (state.currentMistakes || result.mistakes || [])[0];

    let message = score >= 90
        ? phrase("greatSession", lang)
        : score >= 75
            ? phrase("goodSession", lang)
            : phrase("sessionComplete", lang);

    if (topMistake) message += ` ${phrase("topTip", lang, topMistake.title)}`;

    speakText(message, lang);
}

function playVoiceCoaching() {
    if (!("speechSynthesis" in window)) {
        showToast("Voice coaching is not supported by this browser.");
        return;
    }
    if (!state.currentSession) {
        showToast("No analysis available yet.");
        return;
    }

    window.speechSynthesis.cancel();

    const language = state.feedbackLanguage || state.voiceLanguage;
    const queue = [state.currentFeedbackText || buildSpeechFeedback(state.currentSession)];

    (state.currentMistakes || []).slice(0, 4).forEach(mistake => {
        const line = mistake.correction
            ? `${mistake.title}. ${mistake.description}. To fix it: ${mistake.correction}`
            : `${mistake.title}. ${mistake.description}`;
        queue.push(line);
    });

    speakQueue(queue, language, 0);
    showToast("Playing voice coaching...");
}

function speakQueue(messages, language, index) {
    if (index >= messages.length) return;

    const utterance = new SpeechSynthesisUtterance(messages[index]);
    utterance.lang = language || "en-IN";
    utterance.rate = 0.95;
    utterance.onend = () => speakQueue(messages, language, index + 1);
    utterance.onerror = () => speakQueue(messages, language, index + 1);

    window.speechSynthesis.speak(utterance);
}


/* =========================================================
   SENIOR MODE
========================================================= */

function loadSeniorMode() {
    try {
        state.seniorMode = localStorage.getItem("movementCoachSeniorMode") === "true";
    } catch (error) {
        state.seniorMode = false;
    }
}

function setupSeniorMode() {
    applySeniorModeUI();

    $("seniorModeToggle")?.addEventListener("click", toggleSeniorMode);
    $("settingsSeniorToggle")?.addEventListener("click", toggleSeniorMode);
}

function toggleSeniorMode() {
    state.seniorMode = !state.seniorMode;

    try {
        localStorage.setItem("movementCoachSeniorMode", String(state.seniorMode));
    } catch (error) {
        console.warn(error);
    }

    applySeniorModeUI();
    renderSeniorTips(modeData[state.selectedMode]);
    showToast(state.seniorMode ? "Senior Mode turned on." : "Senior Mode turned off.");
}

function applySeniorModeUI() {
    document.body.classList.toggle("senior-mode", state.seniorMode);

    const sidebarToggle = $("seniorModeToggle");
    const settingsToggle = $("settingsSeniorToggle");

    if (sidebarToggle) {
        sidebarToggle.classList.toggle("active", state.seniorMode);
        sidebarToggle.setAttribute("aria-pressed", String(state.seniorMode));
    }

    if (settingsToggle) settingsToggle.textContent = state.seniorMode ? "Turn Off" : "Turn On";
}


/* =========================================================
   VOICE LANGUAGE
========================================================= */

function loadVoiceLanguage() {
    try {
        const saved = localStorage.getItem("movementCoachVoiceLanguage");
        if (saved) {
            state.voiceLanguage = saved;
            state.feedbackLanguage = saved;
        }
    } catch (error) {
        console.warn(error);
    }

    $$("[data-voice-lang]").forEach(button => {
        button.classList.toggle("active", button.dataset.voiceLang === state.voiceLanguage);
    });
}

function setVoiceLanguage(language, persist = true) {
    state.voiceLanguage = language;

    if (persist) {
        try {
            localStorage.setItem("movementCoachVoiceLanguage", language);
        } catch (error) {
            console.warn(error);
        }
    }
}


/* =========================================================
   PERSONALIZED FITNESS PLAN
========================================================= */

function loadSavedPlan() {
    try {
        const saved = localStorage.getItem("movementCoachFitnessPlan");
        state.savedPlan = saved ? JSON.parse(saved) : null;
    } catch (error) {
        state.savedPlan = null;
    }
}

function setupFitnessPlan() {
    $$("#planGoalButtons [data-goal]").forEach(button => {
        button.addEventListener("click", () => {
            $$("#planGoalButtons [data-goal]").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            state.planGoal = button.dataset.goal;
        });
    });

    $$("#planLevelButtons [data-level]").forEach(button => {
        button.addEventListener("click", () => {
            $$("#planLevelButtons [data-level]").forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            state.planLevel = button.dataset.level;
        });
    });

    $("generatePlanBtn")?.addEventListener("click", generateFitnessPlan);

    if (state.savedPlan) renderFitnessPlan(state.savedPlan);
}

async function generateFitnessPlan() {
    const button = $("generatePlanBtn");
    const resultEl = $("planResult");

    if (button) {
        button.disabled = true;
        button.textContent = "Generating plan...";
    }

    resultEl.innerHTML = `<div class="empty-message">🤖 Building your personalized plan...</div>`;

    try {
        const response = await fetch("/api/fitness-plan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                goal: state.planGoal,
                level: state.planLevel,
                mode: state.selectedMode,
                language: state.voiceLanguage
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Backend ${response.status}: ${errorText}`);
        }

        const data = await response.json();

        state.savedPlan = { goal: state.planGoal, level: state.planLevel, createdAt: new Date().toISOString(), ...data };

        try {
            localStorage.setItem("movementCoachFitnessPlan", JSON.stringify(state.savedPlan));
        } catch (error) {
            console.warn(error);
        }

        renderFitnessPlan(state.savedPlan);
        showToast("Fitness plan generated.");

    } catch (error) {
        console.error("Fitness plan failed:", error);
        resultEl.innerHTML = `<div class="empty-message">Could not generate a plan right now. Please check that server.js is running and try again.</div>`;
        showToast("Fitness plan generation failed.");
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = "Generate My Plan →";
        }
    }
}

function renderFitnessPlan(plan) {
    const resultEl = $("planResult");
    if (!resultEl || !plan) return;

    const daily = Array.isArray(plan.daily) ? plan.daily : [];
    const weekly = Array.isArray(plan.weekly) ? plan.weekly : [];

    resultEl.innerHTML = `
        <p class="review-hint" style="padding:0 0 14px;">
            AI Estimate — general fitness guidance for ${escapeHTML(plan.goal || state.planGoal)} ·
            ${escapeHTML(plan.level || state.planLevel)} level. Not a medical program.
        </p>
        <div class="section-title" style="margin-top:0;"><h2 style="font-size:15px;">Daily Plan</h2></div>
        <div class="mistake-list">
            ${daily.length ? daily.map(item => `
                <div class="mistake-card" style="grid-template-columns: 1fr;">
                    <div class="mistake-content"><strong>${escapeHTML(item.title || "Exercise")}</strong><span>${escapeHTML(item.detail || item.description || "")}</span></div>
                </div>
            `).join("") : '<div class="empty-message">No daily items returned.</div>'}
        </div>
        <div class="section-title"><h2 style="font-size:15px;">Weekly Plan</h2></div>
        <div class="mistake-list">
            ${weekly.length ? weekly.map(item => `
                <div class="mistake-card" style="grid-template-columns: 1fr;">
                    <div class="mistake-content"><strong>${escapeHTML(item.day || item.title || "Day")}</strong><span>${escapeHTML(item.detail || item.description || "")}</span></div>
                </div>
            `).join("") : '<div class="empty-message">No weekly items returned.</div>'}
        </div>
        ${plan.recovery ? `
            <div class="section-title"><h2 style="font-size:15px;">Rest &amp; Recovery</h2></div>
            <div class="empty-message" style="text-align:left;">${escapeHTML(plan.recovery)}</div>
        ` : ""}
    `;
}


/* =========================================================
   NEW: ADAPTIVE AI COACH RECOMMENDATION
   Compares this session's metrics with the most recent
   previous session for the SAME exercise, and asks the
   backend (Gemini) for a short personalized recommendation.
========================================================= */

async function fetchCoachRecommendation(session) {
    const card = $("coachRecommendationCard");
    if (!card) return;

    const exercise = session.exercise || session.mode;

    const previous = state.history.find(item =>
        item.id !== session.id && (item.exercise || item.mode) === exercise
    );

    const current = {
        score: session.score,
        reps: session.metrics?.reps || 0,
        correctReps: session.metrics?.correctReps || 0,
        corrections: session.metrics?.corrections || 0,
        duration: Math.round(session.duration || 0)
    };

    const previousPayload = previous ? {
        score: previous.score,
        reps: previous.metrics?.reps || 0,
        correctReps: previous.metrics?.correctReps || 0,
        corrections: previous.metrics?.corrections || 0
    } : null;

    try {
        const response = await fetch("/api/coach-recommendation", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                exercise,
                current,
                previous: previousPayload,
                language: state.voiceLanguage
            })
        });

        if (!response.ok) throw new Error(`Backend ${response.status}`);

        const data = await response.json();
        renderCoachRecommendation(data, exercise);

    } catch (error) {
        console.warn("Coach recommendation failed:", error);
        const lang = state.voiceLanguage || "en-IN";
        renderCoachRecommendation({
            summary: current.score >= 80
                ? phrase("solidSession", lang)
                : phrase("goodEffort", lang),
            recommendation: phrase("keepPracticing", lang),
            focusArea: null
        }, exercise);
    }
}

function renderCoachRecommendation(data, exercise) {
    const card = $("coachRecommendationCard");
    if (!card) return;

    const summary = data.summary || "Session analyzed.";
    const recommendation = data.recommendation || "Keep practicing consistently.";
    const focusArea = data.focusArea;

    card.innerHTML = `
        <div class="coach-avatar">🤖</div>
        <div class="coach-body">
            <strong>${escapeHTML(EXERCISE_LABELS[exercise] || exercise || "This session")}</strong>
            <p>${escapeHTML(summary)}</p>
            <p>${escapeHTML(recommendation)}</p>
            ${focusArea ? `<span class="coach-focus">Focus next time: ${escapeHTML(focusArea)}</span>` : ""}
        </div>
    `;
}


/* =========================================================
   PRIVACY CONTROLS
========================================================= */

function setupPrivacyControls() {
    $("deleteHistoryBtn")?.addEventListener("click", () => {
        if (!confirm("Delete all saved analysis history from this device?")) return;

        state.history = [];
        try { localStorage.removeItem("movementCoachHistory"); } catch (error) { console.warn(error); }

        calculateStats();
        updateProgressUI();
        updateMistakeBadge();
        renderHistoryMistakes();
        showToast("Analysis history deleted.");
    });

    $("deleteNutritionBtn")?.addEventListener("click", () => {
        if (!confirm("Delete all saved nutrition entries from this device?")) return;

        state.nutritionHistory = [];
        try { localStorage.removeItem("movementCoachNutrition"); } catch (error) { console.warn(error); }

        renderNutritionHistory();
        showToast("Nutrition history deleted.");
    });

    $("deletePlanBtn")?.addEventListener("click", () => {
        if (!confirm("Delete your saved fitness plan?")) return;

        state.savedPlan = null;
        try { localStorage.removeItem("movementCoachFitnessPlan"); } catch (error) { console.warn(error); }

        const resultEl = $("planResult");
        if (resultEl) {
            resultEl.innerHTML = `<div class="empty-message">No plan generated yet. Choose a goal and level, then press "Generate My Plan".</div>`;
        }

        showToast("Fitness plan deleted.");
    });
}


/* =========================================================
   NUTRITION HISTORY
========================================================= */

function loadNutritionHistory() {
    try {
        const saved = localStorage.getItem("movementCoachNutrition");
        state.nutritionHistory = saved ? JSON.parse(saved) : [];
    } catch (error) {
        state.nutritionHistory = [];
    }
    renderNutritionHistory();
}

function renderNutritionHistory() {
    const container = $("nutritionHistoryList");
    if (!container) return;

    if (!state.nutritionHistory.length) {
        container.innerHTML = `<div class="empty-message">No nutrition entries yet. Nutrition tracking is not part of this build — once food entries are logged, an "AI Estimate" summary will appear here.</div>`;
        return;
    }

    container.innerHTML = state.nutritionHistory.map(entry => `
        <div class="mistake-card" style="grid-template-columns: 1fr auto;">
            <div class="mistake-content"><strong>${escapeHTML(entry.title || "Food entry")} · AI Estimate</strong><span>${escapeHTML(entry.summary || "")}</span></div>
            <div class="mistake-time">${escapeHTML(entry.date || "")}</div>
        </div>
    `).join("");
}


/* =========================================================
   AI COACH CHATBOT (text + voice input)
   NEW: sends current exercise / form-score context so the
   chatbot behaves as part of one multimodal experience with
   the camera + voice feedback, instead of a separate silo.
========================================================= */

function setupChatWidget() {
    const fab = $("chatFab");
    const panel = $("chatPanel");
    const closeBtn = $("chatCloseBtn");
    const sendBtn = $("chatSendBtn");
    const input = $("chatInput");
    const micBtn = $("chatMicBtn");

    fab?.addEventListener("click", () => panel.classList.toggle("hidden"));
    closeBtn?.addEventListener("click", () => panel.classList.add("hidden"));
    sendBtn?.addEventListener("click", () => sendChatMessage());

    input?.addEventListener("keydown", event => {
        if (event.key === "Enter") sendChatMessage();
    });

    micBtn?.addEventListener("click", toggleChatVoiceInput);
}

function getChatContext() {
    return {
        mode: state.selectedMode,
        exercise: state.activeExercise || state.currentSession?.exercise || null,
        lastFormScore: state.currentSession?.metrics?.avgFormScore ?? null,
        lastOverallScore: state.currentSession?.score ?? null,
        currentPage: state.currentPage
    };
}

async function sendChatMessage() {
    const input = $("chatInput");
    const messagesEl = $("chatMessages");
    const text = (input.value || "").trim();

    if (!text || state.chatBusy) return;

    appendChatBubble(text, "chat-user");
    input.value = "";
    state.chatBusy = true;

    const thinkingBubble = appendChatBubble("Thinking...", "chat-bot");

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                language: state.voiceLanguage,
                context: getChatContext()
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Backend ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        const reply = data.reply || "Sorry, I couldn't generate a reply right now.";

        thinkingBubble.textContent = reply;
        speakText(reply, state.voiceLanguage);

    } catch (error) {
        console.error("Chat failed:", error);
        thinkingBubble.textContent = "The AI coach chat is unavailable. Check that server.js is running.";
        thinkingBubble.classList.add("chat-error");
    } finally {
        state.chatBusy = false;
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }
}

function appendChatBubble(text, className) {
    const messagesEl = $("chatMessages");
    const bubble = document.createElement("div");

    bubble.className = `chat-bubble ${className}`;
    bubble.textContent = text;
    messagesEl.appendChild(bubble);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    return bubble;
}

function toggleChatVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        showToast("Voice input is not supported by this browser.");
        return;
    }

    const micBtn = $("chatMicBtn");

    if (state.isListening) {
        state.speechRecognition?.stop();
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = state.voiceLanguage || "en-IN";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        state.isListening = true;
        micBtn?.classList.add("listening");
    };

    recognition.onresult = event => {
        const transcript = event.results?.[0]?.[0]?.transcript || "";
        const input = $("chatInput");
        if (input) input.value = transcript;
        if (transcript) sendChatMessage();
    };

    recognition.onerror = error => {
        console.warn("Speech recognition error:", error);
        showToast("Could not hear you clearly. Please try again.");
    };

    recognition.onend = () => {
        state.isListening = false;
        micBtn?.classList.remove("listening");
    };

    state.speechRecognition = recognition;

    try { recognition.start(); } catch (error) { console.warn(error); }
}


/* =========================================================
   HISTORY
========================================================= */

function saveSession() {
    if (!state.currentSession) return;

    const historySession = {
        id: state.currentSession.id,
        mode: state.currentSession.mode,
        modeTitle: state.currentSession.modeTitle,
        exercise: state.currentSession.exercise,
        duration: state.currentSession.duration,
        createdAt: state.currentSession.createdAt,
        score: state.currentSession.score,
        breakdown: state.currentSession.breakdown,
        mistakes: state.currentSession.mistakes,
        metrics: state.currentSession.metrics
    };

    state.history.unshift(historySession);
    state.history = state.history.slice(0, 20);

    localStorage.setItem("movementCoachHistory", JSON.stringify(state.history));
    calculateStats();
}

function loadHistory() {
    try {
        const saved = localStorage.getItem("movementCoachHistory");
        state.history = saved ? JSON.parse(saved) : [];
    } catch (error) {
        console.error(error);
        state.history = [];
    }
    calculateStats();
}

function calculateStats() {
    state.sessions = state.history.length;

    state.bestScore = state.history.reduce((best, session) => Math.max(best, Number(session.score) || 0), 0);

    state.totalIssues = state.history.reduce((total, session) =>
        total + (Array.isArray(session.mistakes) ? session.mistakes.length : 0), 0);
}


/* =========================================================
   HISTORY PAGE
========================================================= */

function renderHistoryMistakes() {
    const container = $("historyMistakes");
    const sessionsWithMistakes = state.history.filter(session => Array.isArray(session.mistakes) && session.mistakes.length);

    if (!sessionsWithMistakes.length) {
        container.innerHTML = `<div class="empty-message">No mistakes recorded yet.</div>`;
        return;
    }

    const items = [];
    sessionsWithMistakes.forEach(session => {
        session.mistakes.forEach(mistake => {
            items.push({ ...mistake, mode: session.modeTitle, date: formatDate(session.createdAt) });
        });
    });

    container.innerHTML = items.map(mistake => `
        <div class="mistake-card">
            <div class="mistake-time">${escapeHTML(mistake.timestamp || formatTime(mistake.time))}</div>
            <div class="mistake-icon">!</div>
            <div class="mistake-content">
                <strong>${escapeHTML(mistake.title)}</strong>
                <span>${escapeHTML(mistake.description)} · ${escapeHTML(mistake.mode)} · ${escapeHTML(mistake.date)}</span>
                ${mistake.correction ? `<span class="mistake-fix"><b>${escapeHTML(fixLabelForLanguage())}:</b> ${escapeHTML(mistake.correction)}</span>` : ""}
            </div>
            <div>🔊</div>
        </div>
    `).join("");
}


/* =========================================================
   PROGRESS
========================================================= */

function updateProgressUI() {
    calculateStats();

    $("sessionCount").textContent = state.sessions;
    $("bestScore").textContent = state.bestScore ? `${state.bestScore}%` : "--";
    $("totalIssues").textContent = state.totalIssues;
    $("progressMode").textContent = state.selectedMode ? modeData[state.selectedMode].title : "--";

    updateTodayVsPrevious();
    renderWeeklyChart();
    renderNutritionHistory();
    renderAchievements();
}

function updateTodayVsPrevious() {
    const todayEl = $("todayScore");
    const previousEl = $("previousScore");
    const improvementEl = $("improvementPercent");
    const weeklyCountEl = $("weeklyCount");

    if (!todayEl) return;

    const sorted = [...state.history].sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    const today = sorted[0];
    const previous = sorted[1];

    todayEl.textContent = today && Number.isFinite(today.score) ? `${today.score}%` : "--";
    previousEl.textContent = previous && Number.isFinite(previous.score) ? `${previous.score}%` : "--";

    if (today && previous && Number.isFinite(today.score) && Number.isFinite(previous.score) && previous.score > 0) {
        const improvement = ((today.score - previous.score) / previous.score) * 100;
        improvementEl.textContent = `${improvement > 0 ? "+" : ""}${improvement.toFixed(1)}%`;
    } else {
        improvementEl.textContent = "--";
    }

    const now = Date.now();
    const weekMs = 7 * 24 * 60 * 60 * 1000;
    const weeklyCount = state.history.filter(session => now - new Date(session.createdAt).getTime() <= weekMs).length;

    if (weeklyCountEl) weeklyCountEl.textContent = weeklyCount;
}

function renderWeeklyChart() {
    const canvas = $("weeklyChart");
    const hint = $("chartEmptyHint");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const sorted = [...state.history]
        .filter(session => Number.isFinite(session.score))
        .sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt))
        .slice(-7);

    if (!sorted.length) {
        hint?.classList.remove("hidden");
        return;
    }

    hint?.classList.add("hidden");

    const padding = 30;
    const barGap = 14;
    const barWidth = (width - padding * 2) / sorted.length - barGap;

    const accent = getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#a78bfa";
    const muted = getComputedStyle(document.body).getPropertyValue("--muted").trim() || "#a1a1aa";

    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    sorted.forEach((session, index) => {
        const barHeight = ((height - padding * 2) * Math.max(0, Math.min(100, session.score))) / 100;
        const x = padding + index * (barWidth + barGap);
        const y = height - padding - barHeight;

        ctx.fillStyle = accent;
        ctx.fillRect(x, y, barWidth, barHeight);

        ctx.fillStyle = "#fafafa";
        ctx.font = "12px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`${session.score}%`, x + barWidth / 2, y - 6);

        ctx.fillStyle = muted;
        ctx.font = "10px Inter, sans-serif";
        ctx.fillText(formatDate(session.createdAt).slice(0, 5), x + barWidth / 2, height - padding + 14);
    });
}

function updateMistakeBadge() {
    calculateStats();
    $("mistakeBadge").textContent = state.totalIssues;
}


/* =========================================================
   NEW: ACHIEVEMENTS (lightweight gamification)
========================================================= */

const ACHIEVEMENT_DEFS = [
    { id: "first_session", label: "🎬 First Session", test: (s, h) => h.length === 1 },
    { id: "perfect_score", label: "💯 Perfect Form", test: s => s.score >= 95 },
    { id: "five_reps", label: "🔥 5+ Reps in a Set", test: s => (s.metrics?.reps || 0) >= 5 },
    { id: "ten_sessions", label: "⭐ 10 Sessions Logged", test: (s, h) => h.length >= 10 },
    { id: "clean_set", label: "✅ Zero Corrections Set", test: s => (s.metrics?.reps || 0) > 0 && (s.metrics?.corrections || 0) === 0 }
];

function loadAchievements() {
    try {
        const saved = localStorage.getItem("movementCoachAchievements");
        state.achievements = saved ? JSON.parse(saved) : [];
    } catch (error) {
        state.achievements = [];
    }
}

function checkAndUnlockAchievements(session) {
    const unlockedIds = new Set(state.achievements.map(a => a.id));
    let newlyUnlocked = null;

    for (const def of ACHIEVEMENT_DEFS) {
        if (unlockedIds.has(def.id)) continue;

        try {
            if (def.test(session, state.history)) {
                state.achievements.push({ id: def.id, label: def.label, unlockedAt: new Date().toISOString() });
                newlyUnlocked = def;
                break;
            }
        } catch (error) {
            console.warn(error);
        }
    }

    if (newlyUnlocked) {
        try {
            localStorage.setItem("movementCoachAchievements", JSON.stringify(state.achievements));
        } catch (error) {
            console.warn(error);
        }
        showToast(`Achievement unlocked: ${newlyUnlocked.label}`);
        renderAchievements();
    }
}

function renderAchievements() {
    const grid = $("achievementsGrid");
    if (!grid) return;

    if (!state.achievements.length) {
        grid.innerHTML = `<div class="empty-message">Complete a session to start unlocking achievements.</div>`;
        return;
    }

    grid.innerHTML = state.achievements.map(a => `
        <div class="stat-card"><div class="stat-icon">🏆</div><div><span>Unlocked</span><strong style="font-size:15px;">${escapeHTML(a.label)}</strong></div></div>
    `).join("");
}


/* =========================================================
   NEW: DOWNLOADABLE SESSION REPORT
========================================================= */

function setupReportDownload() {
    $("downloadReportBtn")?.addEventListener("click", downloadSessionReport);
}

function downloadSessionReport() {
    const session = state.currentSession;
    if (!session) {
        showToast("No session to export yet.");
        return;
    }

    const lines = [
        "MOVEMENTCOACH — SESSION REPORT",
        "================================",
        `Date: ${new Date(session.createdAt).toLocaleString()}`,
        `Mode: ${session.modeTitle}`,
        `Exercise: ${EXERCISE_LABELS[session.exercise] || session.exercise || "-"}`,
        `Duration: ${formatTime(session.duration)}`,
        "",
        `Overall Score: ${session.score}%`,
        `Left: ${session.breakdown?.left ?? "-"}%  Right: ${session.breakdown?.right ?? "-"}%`,
        `Upper: ${session.breakdown?.upper ?? "-"}%  Lower: ${session.breakdown?.lower ?? "-"}%`,
        "",
        `Reps: ${session.metrics?.reps ?? "-"}`,
        `Correct Reps: ${session.metrics?.correctReps ?? "-"}`,
        `Corrections: ${session.metrics?.corrections ?? "-"}`,
        `Avg Live Form Score: ${session.metrics?.avgFormScore ?? "-"}`,
        "",
        "AI Feedback:",
        session.originalFeedback || "-",
        "",
        "Mistakes:",
        ...(session.mistakes || []).map((m, i) => `${i + 1}. [${m.timestamp}] ${m.title} — ${m.description}${m.correction ? ` | Fix: ${m.correction}` : ""}`)
    ];

    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `movementcoach-report-${session.id}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
    showToast("Session report downloaded.");
}


/* =========================================================
   PRACTICE AGAIN
========================================================= */

function practiceAgain() {
    cleanupStudio();
    showPage("modeSetup");
}


/* =========================================================
   BACKEND CHECK
========================================================= */

async function checkBackend() {
    try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error("Backend unavailable");

        const data = await response.json();

        $("aiStatus").textContent = "AI Coach Ready";
        $("aiStatusText").textContent = data.message || "Gemini backend is connected.";
        $("statusDot").classList.add("ready");

        /* ---- NEW: reflect real Gemini configuration state, not a
           hardcoded "Checking..." label ---- */
        state.geminiAvailable = Boolean(data.success);
        const geminiConfigured = !String(data.message || "").toLowerCase().includes("missing");
        setGeminiStatusPills(geminiConfigured ? "● Ready" : "● Not configured");

    } catch (error) {
        console.warn("Backend check failed:", error);
        $("aiStatus").textContent = "Backend Offline";
        $("aiStatusText").textContent = "Start server.js to enable AI analysis.";
        state.geminiAvailable = false;
        setGeminiStatusPills("● Offline");
    }
}

/* ---- NEW: shared helper — keeps the dashboard pill and the
   in-studio live bar pill in sync with the real Gemini state
   instead of two independent hardcoded strings. ---- */
function setGeminiStatusPills(text) {
    const dash = $("dashGeminiStatus");
    const live = $("geminiLiveStatus");
    if (dash) dash.textContent = text;
    if (live) live.textContent = text;
}


/* =========================================================
   RESULT TEXT
========================================================= */

function generateResultText(score, issueCount) {
    if (score >= 90) {
        return `Excellent movement quality. You maintained strong consistency during the session${issueCount ? ` with ${issueCount} areas to refine.` : "."}`;
    }
    if (score >= 80) {
        return `Good progress. Your movement is developing well. Review the ${issueCount} highlighted ${issueCount === 1 ? "moment" : "moments"} to improve consistency.`;
    }
    if (score >= 65) {
        return "Your movement shows progress, but several areas need attention. Review the highlighted timestamps and practice those movements again.";
    }
    return "Keep practicing. Focus on the highlighted moments and improve your movement control step by step.";
}


/* =========================================================
   UTILITIES
========================================================= */

function formatTime(seconds) {
    seconds = Math.max(0, Math.floor(Number(seconds) || 0));
    const minutes = Math.floor(seconds / 60);
    const remaining = seconds % 60;
    return String(minutes).padStart(2, "0") + ":" + String(remaining).padStart(2, "0");
}

function formatDate(date) {
    try {
        return new Date(date).toLocaleDateString();
    } catch {
        return "";
    }
}

function clampScore(score) {
    if (!Number.isFinite(score)) return 0;
    return Math.min(100, Math.max(0, Math.round(score)));
}

function escapeHTML(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


/* =========================================================
   TOAST
========================================================= */

let toastTimeout = null;

function showToast(message) {
    const toast = $("toast");
    if (!toast) return;

    $("toastMessage").textContent = message;
    toast.classList.add("show");

    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => toast.classList.remove("show"), 3500);
}