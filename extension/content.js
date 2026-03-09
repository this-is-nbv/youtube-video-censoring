console.log("✅ Content script injected");

// -------------------------------
// Helpers
// -------------------------------
function getVideo() {
  return document.querySelector("video");
}

function getVideoId() {
  const url = new URL(window.location.href);
  return url.searchParams.get("v");
}

function isAdPlaying() {
  return document.body.classList.contains("ad-showing");
}

// -------------------------------
// Overlay UI
// -------------------------------
let overlay = null;

function showOverlay() {
  if (overlay || isAdPlaying()) return;

  overlay = document.createElement("div");
  overlay.innerText = "🔎 Censoring content… Please wait";

  overlay.style.position = "fixed";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.width = "100vw";
  overlay.style.height = "100vh";
  overlay.style.background = "rgba(0,0,0,0.85)";
  overlay.style.color = "#fff";
  overlay.style.fontSize = "24px";
  overlay.style.display = "flex";
  overlay.style.alignItems = "center";
  overlay.style.justifyContent = "center";
  overlay.style.zIndex = "999999";
  overlay.style.fontFamily = "sans-serif";

  document.body.appendChild(overlay);
}

function removeOverlay() {
  if (overlay) {
    overlay.remove();
    overlay = null;
  }
}

// -------------------------------
// Beep sound
// -------------------------------
let audioCtx = null;
let oscillator = null;

function startBeep() {
  if (oscillator) return;

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  oscillator = audioCtx.createOscillator();
  oscillator.type = "sine";
  oscillator.frequency.setValueAtTime(1000, audioCtx.currentTime);
  oscillator.connect(audioCtx.destination);
  oscillator.start();
}

function stopBeep() {
  if (!oscillator) return;

  oscillator.stop();
  oscillator.disconnect();
  oscillator = null;

  audioCtx.close();
  audioCtx = null;
}

// -------------------------------
// Profanity state
// -------------------------------
let profanityWindows = [];
let profaneWords = [];

let censorActive = false;
let analysisComplete = false;
let monitorInterval = null;

// -------------------------------
// Word masking helper
// -------------------------------
function maskWord(word) {
  return "*".repeat(word.length);
}

// -------------------------------
// Caption censoring
// -------------------------------
function censorCaptions() {

  if (!profaneWords.length) return;

  const captions = document.querySelectorAll(".ytp-caption-segment");

  captions.forEach(node => {

    let text = node.textContent;

    profaneWords.forEach(word => {

      const regex = new RegExp(`\\b${word}\\b`, "gi");

      text = text.replace(regex, maskWord(word));

    });

    node.textContent = text;

  });

}

// -------------------------------
// Title censoring
// -------------------------------
function censorTitle() {

  if (!profaneWords.length) return;

  const title = document.querySelector("h1.ytd-watch-metadata");

  if (!title) return;

  let text = title.textContent;

  profaneWords.forEach(word => {

    const regex = new RegExp(`\\b${word}\\b`, "gi");

    text = text.replace(regex, maskWord(word));

  });

  title.textContent = text;
}

// -------------------------------
// Strong pause guard
// -------------------------------
let pauseGuard = null;

function startPauseGuard(video) {

  pauseGuard = setInterval(() => {

    if (!analysisComplete && !isAdPlaying()) {
      video.pause();
      video.muted = true;
    }

  }, 100);
}

function stopPauseGuard() {
  clearInterval(pauseGuard);
}

// -------------------------------
// Profanity monitoring
// -------------------------------
function monitorPlayback() {

  const video = getVideo();
  if (!video || isAdPlaying()) return;

  const t = video.currentTime;

  const inProfanity = profanityWindows.some(
    w => t >= w.start && t <= w.end
  );

  if (inProfanity && !censorActive) {

    video.muted = true;
    startBeep();

    censorActive = true;
  }

  if (!inProfanity && censorActive) {

    video.muted = false;
    stopBeep();

    censorActive = false;
  }
}

// -------------------------------
// Reset everything for new video
// -------------------------------
function resetState() {

  console.log("🔄 Resetting censor state");

  profanityWindows = [];
  profaneWords = [];

  censorActive = false;
  analysisComplete = false;

  stopBeep();
  stopPauseGuard();
  removeOverlay();

  if (monitorInterval) {
    clearInterval(monitorInterval);
    monitorInterval = null;
  }
}

// -------------------------------
// Main analysis flow
// -------------------------------
function initCensoring(videoId) {

  const video = getVideo();
  if (!video) return;

  console.log("🧠 Analyzing video:", videoId);

  resetState();
  showOverlay();
  startPauseGuard(video);

  chrome.runtime.sendMessage(
    {
      type: "ANALYZE_VIDEO",
      payload: { video_id: videoId }
    },
    (response) => {

      analysisComplete = true;

      stopPauseGuard();
      removeOverlay();

      if (!response || !response.success) {

        console.error("❌ Backend error:", response?.error);

        video.muted = false;
        return;
      }

      // NEW: get normalized data
      profanityWindows = response.data.windows || [];
      profaneWords = response.data.words || [];

      console.log("🚫 Profanity windows:", profanityWindows);
      console.log("🚫 Profane words:", profaneWords);

      // Apply title censor once
      censorTitle();

      video.muted = false;
      video.play();

      monitorInterval = setInterval(() => {

        monitorPlayback();
        censorCaptions();

      }, 100);

    }
  );
}

// -------------------------------
// Detect video changes (SPA-safe)
// -------------------------------
let currentVideoId = null;

setInterval(() => {

  const newVideoId = getVideoId();
  const video = getVideo();

  if (!video || isAdPlaying()) return;

  if (newVideoId && newVideoId !== currentVideoId) {

    console.log("🎬 Video changed:", newVideoId);

    currentVideoId = newVideoId;

    initCensoring(newVideoId);
  }

}, 500);