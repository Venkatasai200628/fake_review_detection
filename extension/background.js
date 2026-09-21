// background.js -- the only part that talks to the backend.
// Content scripts run inside amazon.in / flipkart.com / meesho.com, where the page's own
// security rules could block a request to the backend. The service worker has
// host_permissions, so it makes the call and passes the answer back.
//
// The backend is normally http://127.0.0.1:8000 on this computer. It can also be a hosted
// copy (Codespaces, Azure, a container host) -- set the URL and token in the extension
// popup. Two things to know about a hosted one:
//   * it must be https. Chrome will not let the extension call plain http from an https page.
//   * set RPF_TOKEN on the server and paste the same token in the popup, otherwise anyone
//     who finds the URL can use your server (deploy/README.md).

const DEFAULT_BACKEND = 'http://127.0.0.1:8000';

async function settings() {
  const { backendUrl, backendToken } = await chrome.storage.local.get(['backendUrl', 'backendToken']);
  return {
    url: (backendUrl || DEFAULT_BACKEND).replace(/\/+$/, ''),
    token: backendToken || '',
  };
}

function headers(token) {
  const h = { 'Content-Type': 'application/json' };
  if (token) h.Authorization = `Bearer ${token}`;
  return h;
}

/** Turn a failure into something the card can actually show the user. */
function explain(res, url) {
  if (res.status === 401) return 'backend rejected the token: check it in the extension popup';
  if (res.status === 403) return 'backend refused the request (CORS or token)';
  return `backend HTTP ${res.status} from ${url}`;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'analyze') {
    settings().then(({ url, token }) =>
      fetch(`${url}/analyze`, {
        method: 'POST',
        headers: headers(token),
        body: JSON.stringify(msg.payload),
      })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(explain(r, url)))))
        .then((data) => {
          chrome.storage.local.set({
            lastSummary: { ...data.summary, url: msg.payload.page_url, at: Date.now() },
          });
          sendResponse({ ok: true, data });
        })
        .catch((e) => sendResponse({ ok: false, error: String(e.message || e) })));
    return true; // keep the channel open for the async answer
  }

  if (msg.type === 'health') {
    settings().then(({ url, token }) =>
      fetch(`${url}/health`, { headers: headers(token) })
        .then((r) => r.json())
        .then((data) => sendResponse({ ok: true, data, url }))
        .catch((e) => sendResponse({ ok: false, error: String(e.message || e), url })));
    return true;
  }

  return false;
});
