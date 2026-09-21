const $ = (id) => document.getElementById(id);

function refresh() {
  $('dot').className = 'dot';
  $('status').textContent = 'Checking backend…';
  chrome.runtime.sendMessage({ type: 'health' }, (resp) => {
    // Show WHICH backend answered (or failed), so a wrong address is obvious rather than
    // looking like the server is down.
    $('port').textContent = (resp?.url || '').replace(/^https?:\/\//, '') || '–';
    if (resp && resp.ok) {
      $('dot').className = 'dot ok';
      $('status').textContent = 'Backend running';
      $('features').textContent = `${resp.data.features}${resp.data.uses_cnn ? ' + CNN' : ''}`;
      $('memory').textContent = resp.data.memory_size.toLocaleString();
      const remote = !/127\.0\.0\.1|localhost/.test(resp.url || '');
      $('foot').textContent = remote
        ? (resp.data.auth_required
          ? 'Hosted backend, token required. Photos are sent to that server.'
          : 'Hosted backend with NO token: anyone with the address can use it.')
        : 'Everything runs on this computer. Nothing is uploaded.';
    } else {
      $('dot').className = 'dot bad';
      $('status').textContent = 'Backend not running';
      $('features').textContent = '–';
      $('memory').textContent = '–';
      if (resp?.error) $('saved').textContent = resp.error;
    }
  });
}

refresh();

chrome.storage.local.get(
  ['enabled', 'lastSummary', 'backendUrl', 'backendToken'],
  ({ enabled = true, lastSummary, backendUrl = '', backendToken = '' }) => {
    $('enabled').checked = enabled;
    $('backendUrl').value = backendUrl;
    $('backendToken').value = backendToken;
    if (backendUrl) $('adv').open = true;
    if (!lastSummary) return;
    const box = $('last');
    box.className = 'last';
    box.replaceChildren();
    [['fake', lastSummary.fake, 'likely fake'],
     ['verify', lastSummary.needs_verification, 'verify'],
     ['ok', lastSummary.genuine, 'genuine']].forEach(([cls, n, label]) => {
      const c = document.createElement('div');
      c.className = `chip ${cls}`;
      const b = document.createElement('b');
      b.textContent = n;
      c.append(b, label);
      box.append(c);
    });
  });

$('enabled').addEventListener('change', (e) => chrome.storage.local.set({ enabled: e.target.checked }));

$('save').addEventListener('click', async () => {
  const url = $('backendUrl').value.trim().replace(/\/+$/, '');
  const token = $('backendToken').value.trim();
  if (url && !/^https:\/\//.test(url) && !/^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(url)) {
    // Chrome blocks plain http from an https page, so a remote http address would fail on
    // every real product page and look like a broken backend.
    $('saved').textContent = 'A remote backend must start with https:// (http is only allowed for 127.0.0.1).';
    return;
  }
  await chrome.storage.local.set({ backendUrl: url, backendToken: token });
  $('saved').textContent = url ? 'Saved. Reload the product page.' : 'Using this computer again.';
  refresh();
});
