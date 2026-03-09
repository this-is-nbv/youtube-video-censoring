// ======================================
// Background Service Worker (MV3)
// Handles all backend communication
// ======================================

console.log("🟢 Background service worker started");

// --------------------------------------------------
// Listen for messages from content scripts
// --------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  console.log("📩 Message received in background:", message);

  if (message.type !== "ANALYZE_VIDEO") {
    console.warn("⚠️ Unknown message type:", message.type);
    return;
  }

  const payload = message.payload;

  if (!payload || !payload.video_id) {

    console.error("❌ Missing video_id in payload");

    sendResponse({
      success: false,
      error: "Missing video_id"
    });

    return;
  }

  console.log("🚀 Sending request to backend:", payload);

  fetch("http://127.0.0.1:8000/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  })
    .then(response => {

      console.log("📡 Backend HTTP status:", response.status);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      return response.json();
    })

    .then(data => {

      console.log("📤 Backend response received:", data);

      // Normalize backend output
      const formatted = {
        label: data.label || "SAFE",
        windows: data.profanity_windows || [],
        words: data.profane_words || []
      };

      sendResponse({
        success: true,
        data: formatted
      });

    })

    .catch(error => {

      console.error("❌ Backend fetch failed:", error);

      sendResponse({
        success: false,
        error: error.toString()
      });

    });

  // Required for async response
  return true;
});