// content.js -- runs inside the product page.
// 1. finds every review card (selectors from sites.js)
// 2. reads text, stars, date, reviewer, photo URLs (+ the product's own listing photos)
// 3. sends them to the backend through background.js
// 4. draws a verdict card on each review and a summary panel on top
// Read-only: it never clicks, posts or stores anything from the page.

(() => {
  const site = currentSite();
  if (!site) return;

  const VERDICT = {
    Fake: { cls: 'rpf-fake', label: 'Likely fake', icon: 'x' },
    Needs_Verification: { cls: 'rpf-verify', label: 'Needs verification', icon: 'alert' },
    Genuine: { cls: 'rpf-ok', label: 'Looks genuine', icon: 'check' },
  };
  const ICON = {
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
    alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
    x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
    lens: '<svg viewBox="0 0 24 24" fill="none" stroke="#fdb022" stroke-width="2.4" stroke-linecap="round"><rect x="3" y="4" width="13" height="11" rx="2" stroke="#fff"/><circle cx="15" cy="14" r="4"/><path d="m18 17 3 3"/></svg>',
  };
  const NEUTRAL = /can be normal|no specific|no photo|less reliable/;
  let busy = false;
  let FEEDBACK = {};        // reviewKey -> 'Genuine' | 'Needs_Verification' | 'Fake'
  let LAST = null;          // last analysed cards, so a click can repaint without re-analysing

  const txt = (el, sel) => (el.querySelector(sel)?.innerText || '').trim();
  const absUrl = (u) => (u ? new URL(u, location.href).href : null);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  };
  const level = (p) => (p >= 0.65 ? 'rpf-hi' : p >= 0.35 ? 'rpf-mid' : 'rpf-lo');

  // A site either exposes findCards()/parse() (when its class names are build-generated
  // and selectors cannot be trusted -- Flipkart) or plain CSS selectors (Amazon).
  const findCards = () => (site.findCards ? site.findCards() : [...document.querySelectorAll(site.review)]);

  function fields(c) {
    if (site.parse) {
      const r = site.parse(c);
      return { ...r, images: (r.images || []).map((u) => absUrl(site.fullSize(u))).filter(Boolean) };
    }
    return {
      text: txt(c, site.text),
      rating: txt(c, site.rating) || c.querySelector(site.rating)?.getAttribute('aria-label') || '',
      date: txt(c, site.date),
      reviewer: txt(c, site.author),
      images: [...c.querySelectorAll(site.images)]
        .map((img) => absUrl(site.fullSize(img.getAttribute('data-src') || img.src)))
        .filter(Boolean),
    };
  }

  // ---------------------------------------------------- your own verdict
  // A review is remembered by WHAT IT SAYS, not by where it sat on the page: Amazon and
  // Flipkart reorder reviews, paginate them and hand out different image URLs for the same
  // photo, so anything positional would lose your mark on the next visit.
  function reviewKey(rev) {
    const t = (rev.text || '').replace(/\s+/g, ' ').trim().toLowerCase().slice(0, 400);
    const s = `${(rev.reviewer || '').trim().toLowerCase()}|${t}`;
    let h1 = 5381, h2 = 52711;
    for (let i = 0; i < s.length; i++) {
      h1 = (h1 * 33) ^ s.charCodeAt(i);
      h2 = (h2 * 33) ^ s.charCodeAt(s.length - 1 - i);
    }
    return ((h1 >>> 0).toString(36) + (h2 >>> 0).toString(36));
  }

  const USER_RISK = { Genuine: 0, Needs_Verification: 0.5, Fake: 1 };

  async function saveVerdict(key, value) {
    const { feedback = {} } = await chrome.storage.local.get('feedback');
    if (value) feedback[key] = value; else delete feedback[key];
    await chrome.storage.local.set({ feedback });
    FEEDBACK = feedback;
  }

  function collect() {
    const cards = findCards().filter((c) => !c.dataset.rpfDone);
    const reviews = cards.map((c, i) => {
      const id = c.id || `rpf-${Date.now()}-${i}`;
      c.dataset.rpfId = id;
      const f = fields(c);
      c.dataset.rpfKey = reviewKey(f);
      return { id, ...f };
    });
    return { cards, reviews };
  }

  function listingImages() {
    return [...document.querySelectorAll(site.listingImages)]
      .map((img) => absUrl(img.getAttribute('data-old-hires') || img.src))
      .filter((u) => u && !u.startsWith('data:'))
      .slice(0, 6);
  }

  // ---------------------------------------------------------------- cards
  function placeholder(card) {
    card.querySelector(':scope > .rpf-card')?.remove();
    const c = el('div', 'rpf-card rpf-loading');
    const head = el('div', 'rpf-head');
    head.append(el('div', 'rpf-spin'), el('span', null, 'Checking photo and text…'));
    const body = el('div', 'rpf-body');
    body.append(el('div', 'rpf-skel'), el('div', 'rpf-skel'));
    body.lastChild.style.width = '60%';
    c.append(head, body);
    card.prepend(c);
  }

  function meter(name, p) {
    const m = el('div', 'rpf-meter' + (p == null ? ' rpf-na' : ''));
    const lab = el('label');
    lab.append(el('span', null, name), el('b', null, p == null ? 'no photo' : `${Math.round(p * 100)}%`));
    const track = el('div', 'rpf-track');
    const fill = el('div', `rpf-fill ${p == null ? '' : level(p)}`);
    track.append(fill);
    m.append(lab, track);
    requestAnimationFrame(() => { fill.style.width = `${Math.round((p || 0) * 100)}%`; });
    return m;
  }

  function verdictCard(card, r) {
    card.querySelector(':scope > .rpf-card')?.remove();

    // Your verdict wins over the model's. The model's own score is kept and shown underneath
    // rather than thrown away: you should be able to see what you overruled, and it is the
    // record a future version would learn from.
    const key = card.dataset.rpfKey;
    const mine = FEEDBACK[key];
    const modelRisk = r.risk;
    const modelDecision = r.decision;
    if (mine) r = { ...r, risk: USER_RISK[mine], decision: mine };

    const v = VERDICT[r.decision] || VERDICT.Needs_Verification;
    const c = el('div', `rpf-card ${v.cls}`);

    const head = el('div', 'rpf-head');
    const pill = el('span', 'rpf-pill');
    pill.innerHTML = ICON[v.icon];                 // static SVG only
    pill.append(document.createTextNode(v.label));
    const risk = el('div', 'rpf-risk');
    risk.append(el('b', null, `${Math.round(r.risk * 100)}%`), el('span', null, 'fake risk'));
    head.append(pill, risk);

    const body = el('div', 'rpf-body');
    const meters = el('div', 'rpf-meters');
    meters.append(meter('Text evidence', r.text_score), meter('Photo evidence', r.image_score));
    const ul = el('ul', 'rpf-reasons');
    (r.reasons.length ? r.reasons : ['no specific warning signs']).forEach((t) =>
      ul.append(el('li', NEUTRAL.test(t) ? 'rpf-neutral' : '', t)));
    body.append(meters, ul);

    // The evidence behind "this photo was seen before", so the claim can be checked
    // instead of taken on trust: who posted it, when, and how close the match is.
    if (r.matches && r.matches.length) {
      const det = el('details', 'rpf-evi');
      det.append(el('summary', null, `where this photo was seen before (${r.matches.length})`));
      const t = el('table');
      const hr = el('tr');
      ['account', 'date', 'closeness', 'source'].forEach((h) => hr.append(el('th', null, h)));
      t.append(hr);
      r.matches.forEach((m) => {
        const tr = el('tr');
        const close = m.visual_similarity != null
          ? `${Math.round(m.visual_similarity * 100)}% alike`
          : `hash distance ${m.phash_distance}`;
        [m.reviewer, m.date, close, m.source].forEach((x) => tr.append(el('td', null, String(x))));
        t.append(tr);
      });
      det.append(t);
      body.append(det);
    }

    // ---- your verdict: three buttons, applied instantly and remembered
    const fb = el('div', 'rpf-fb');
    fb.append(el('span', 'rpf-fb-label', mine ? 'Marked by you:' : 'Do you agree?'));
    const btns = el('div', 'rpf-fb-btns');
    [['Genuine', 'Genuine'], ['Needs_Verification', 'Not sure'], ['Fake', 'Fake']]
      .forEach(([val, label]) => {
        const b = el('button', `rpf-fb-btn ${VERDICT[val].cls}${mine === val ? ' on' : ''}`, label);
        b.type = 'button';
        b.addEventListener('click', async (e) => {
          e.preventDefault();
          e.stopPropagation();
          await saveVerdict(key, mine === val ? null : val);   // clicking the same one clears it
          verdictCard(card, { ...r, risk: modelRisk, decision: modelDecision });
          if (LAST) summaryPanel(countWithFeedback(LAST.summary, LAST.cards, LAST.byId));
        });
        btns.append(b);
      });
    fb.append(btns);
    body.append(fb);
    if (mine) {
      body.append(el('div', 'rpf-fb-note',
        `You marked this "${VERDICT[mine].label}". The model said ${Math.round(modelRisk * 100)}% `
        + `(${VERDICT[modelDecision].label}). Click the same button again to undo.`));
    }

    body.append(el('div', 'rpf-foot', 'Review Photo Forensics · checked on this computer'));

    c.append(head, body);
    card.prepend(c);
    card.dataset.rpfDone = '1';
  }

  // -------------------------------------------------------------- summary
  function summaryPanel(summary, error) {
    let s = document.getElementById('rpf-summary');
    if (!s) {
      s = el('div');
      s.id = 'rpf-summary';
      const first = site.review ? document.querySelector(site.review) : findCards()[0];
      (first?.parentElement || document.body).prepend(s);
    }
    s.replaceChildren();
    s.className = error ? 'rpf-error' : '';

    const top = el('div', 'rpf-s-top');
    const logo = el('div', 'rpf-logo');
    logo.innerHTML = ICON.lens;                    // static SVG only
    const t = el('div');
    t.append(el('div', 'rpf-title', error ? 'Review check unavailable' : 'Review Photo Forensics'),
      el('div', 'rpf-sub', error ? error
        : `${summary.checked} reviews checked · photos, text and posting patterns`));
    top.append(logo, t);
    s.append(top);

    if (error) {
      const n = el('div', 'rpf-note');
      n.append('Start the backend: ', el('code', null, 'python -m uvicorn app:app --port 8000'));
      s.append(n);
      return;
    }
    const chips = el('div', 'rpf-chips');
    const stack = el('div', 'rpf-stack');
    [['rpf-fake', summary.fake, 'likely fake'],
     ['rpf-verify', summary.needs_verification, 'need verification'],
     ['rpf-ok', summary.genuine, 'look genuine']].forEach(([cls, n, lab]) => {
      const chip = el('span', `rpf-chip ${cls}`);
      chip.append(el('b', null, String(n)), lab);
      chips.append(chip);
      if (n) {
        const seg = el('i', cls);
        seg.style.flex = String(n);
        stack.append(seg);
      }
    });
    s.append(chips, stack,
      el('div', 'rpf-note', 'A verdict is a signal, not proof. Amber means the evidence is mixed.'));
  }

  // ------------------------------------------------------------------ run
  async function run() {
    if (busy) return;
    const { enabled = true } = await chrome.storage.local.get('enabled');
    if (!enabled) return;
    const { cards, reviews } = collect();
    if (!reviews.length) return;
    busy = true;
    cards.forEach(placeholder);
    const payload = {
      page_url: location.href,
      product_id: site.productId(),
      product_title: (document.querySelector(site.title)?.innerText || '').trim(),
      listing_images: listingImages(),
      reviews,
    };
    chrome.runtime.sendMessage({ type: 'analyze', payload }, (resp) => {
      busy = false;
      if (!resp || !resp.ok) {
        cards.forEach((c) => c.querySelector(':scope > .rpf-card')?.remove());
        summaryPanel(null, resp ? resp.error : 'no answer from the extension');
        return;
      }
      const byId = Object.fromEntries(resp.data.reviews.map((r) => [r.id, r]));
      cards.forEach((c) => byId[c.dataset.rpfId] && verdictCard(c, byId[c.dataset.rpfId]));
      LAST = { cards, byId, summary: resp.data.summary };
      summaryPanel(countWithFeedback(resp.data.summary, cards, byId));
    });
  }

  // The tally at the top has to agree with the cards under it, so a review you re-marked is
  // counted the way YOU marked it.
  function countWithFeedback(summary, cards, byId) {
    const out = { ...summary, fake: 0, needs_verification: 0, genuine: 0 };
    const bucket = { Fake: 'fake', Needs_Verification: 'needs_verification', Genuine: 'genuine' };
    cards.forEach((c) => {
      const r = byId[c.dataset.rpfId];
      if (!r) return;
      const d = FEEDBACK[c.dataset.rpfKey] || r.decision;
      out[bucket[d]] += 1;
    });
    return out;
  }

  function repaint() {
    if (!LAST) return;
    LAST.cards.forEach((c) => {
      const r = LAST.byId[c.dataset.rpfId];
      if (r) verdictCard(c, r);
    });
    summaryPanel(countWithFeedback(LAST.summary, LAST.cards, LAST.byId));
  }

  function clearAll() {
    document.querySelectorAll('.rpf-card').forEach((c) => c.remove());
    document.getElementById('rpf-summary')?.remove();
    document.querySelectorAll('[data-rpf-done]').forEach((c) => delete c.dataset.rpfDone);
  }

  // The switch in the popup takes effect straight away, on the page you are already looking at:
  // turning it off strips every badge, turning it back on re-checks. Otherwise the only way to
  // stop it would be chrome://extensions, which is not where anyone wants to go mid-shopping.
  chrome.storage.onChanged.addListener((ch, area) => {
    if (area !== 'local') return;
    if (ch.feedback) {                 // marked in another tab: keep this page in step
      FEEDBACK = ch.feedback.newValue || {};
      repaint();
    }
    if (!ch.enabled) return;
    clearAll();
    if (ch.enabled.newValue) run();
  });

  // Load your saved verdicts BEFORE the first paint, so a reload shows your marks and not the
  // model's, which is the whole point of remembering them.
  chrome.storage.local.get('feedback').then(({ feedback = {} }) => {
    FEEDBACK = feedback;
    run();
  });
  let t = null;
  new MutationObserver((muts) => {
    // ignore changes we made ourselves
    if (muts.every((m) => [...m.addedNodes].every((n) => n.nodeType === 1 &&
        (n.classList?.contains('rpf-card') || n.id === 'rpf-summary' || n.closest?.('.rpf-card, #rpf-summary'))))) return;
    clearTimeout(t);
    t = setTimeout(run, 1200);
  }).observe(document.body, { childList: true, subtree: true });
})();
