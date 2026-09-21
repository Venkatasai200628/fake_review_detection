// background.js -- the only part that talks to the local backend.
// Content scripts run inside amazon.in / flipkart.com, where the page's own
// security rules could block a request to 127.0.0.1. The service worker has
// host_permissions for the backend, so it makes the call and passes the answer back.

const BACKEND = 'http://127.0.0.1:8000';

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'analyze') {
    fetch(`${BACKEND}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(msg.payload),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`backend HTTP ${r.status}`))))
      .then((data) => {
        chrome.storage.local.set({ lastSummary: { ...data.summary, url: msg.payload.page_url, at: Date.now() } });
        sendResponse({ ok: true, data });
      })
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true; // keep the channel open for the async answer
  }
  if (msg.type === 'health') {
    fetch(`${BACKEND}/health`)
      .then((r) => r.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((e) => sendResponse({ ok: false, error: String(e) }));
    return true;
  }
});
