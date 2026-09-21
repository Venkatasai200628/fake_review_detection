const $ = (id) => document.getElementById(id);

chrome.runtime.sendMessage({ type: 'health' }, (resp) => {
  if (resp && resp.ok) {
    $('dot').className = 'dot ok';
    $('status').textContent = 'Backend running';
    $('features').textContent = `${resp.data.features}${resp.data.uses_cnn ? ' + CNN' : ''}`;
    $('memory').textContent = resp.data.memory_size.toLocaleString();
  } else {
    $('dot').className = 'dot bad';
    $('status').textContent = 'Backend not running';
  }
});

chrome.storage.local.get(['enabled', 'lastSummary'], ({ enabled = true, lastSummary }) => {
  $('enabled').checked = enabled;
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
