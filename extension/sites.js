// sites.js -- WHERE the reviews are on each site. When Amazon or Flipkart change
// their HTML, only this file needs editing (guide section 19.4).
//
// Two ways to describe a site:
//   (a) CSS selectors        - works where the site uses stable, meaningful attributes (Amazon)
//   (b) findCards() + parse() - works where class names are build-generated garbage (Flipkart)
// content.js uses (b) when a site provides it, otherwise (a).
//
// Verified against live pages on 20 September 2026 (see readmes/readme13.md).

const SITES = {
  amazon: {
    // Amazon review cards carry data-hook attributes that have been stable for years
    review: '[data-hook="review"]',
    // CHECKED 20 Sep 2026: amazon.in no longer emits data-hook="review-body" on the
    // product page; the body is data-hook="reviewText" (inside "reviewTextContainer").
    // The old hook is kept first so amazon.com pages that still use it keep working.
    text: '[data-hook="review-body"], [data-hook="reviewText"], [data-hook="reviewTextContainer"]',
    rating: '[data-hook="review-star-rating"], [data-hook="cmps-review-star-rating"]',
    date: '[data-hook="review-date"]',
    author: '.a-profile-name',
    images: 'img[data-hook="review-image-tile"], img.review-image-tile',
    title: '#productTitle',
    listingImages: '#landingImage, #imgTagWrapperId img, #altImages img',
    productId: () => (location.pathname.match(/\/(?:dp|product-reviews|gp\/product)\/([A-Z0-9]{10})/) || [])[1],
    // thumbnails look like .../I/abc123._SY88_.jpg -> drop the size code to get the full photo
    fullSize: (src) => src.replace(/\._[^./]+_\./, '.'),
  },

  flipkart: {
    // Flipkart ships hashed, per-build class names ("v1zwn21n _1psv1zeb9 css-146c3p1"),
    // so selectors written against them break on the next deploy. CHECKED 20 Sep 2026.
    // Instead a review card is found by the one thing that is stable: every real review
    // carries a buyer badge, and the card starts with the star rating.
    BUYER_MARK: /Verified Purchase|Certified Buyer/,

    findCards() {
      // innermost element carrying the badge, then walk up until the block also holds the rating
      const seeds = [...document.querySelectorAll('div, span')].filter(
        (e) => SITES.flipkart.BUYER_MARK.test(e.innerText || '')
          && ![...e.children].some((c) => SITES.flipkart.BUYER_MARK.test(c.innerText || ''))
      );
      const cards = [];
      for (const s of seeds) {
        let p = s;
        for (let i = 0; i < 8 && p.parentElement; i++) {
          p = p.parentElement;
          const t = p.innerText || '';
          if (t.length > 120 && /(^|\n)[1-5](\.\d)?\n/.test(t)) break;
        }
        if (p && !cards.includes(p)) cards.push(p);
      }
      return cards;
    },

    parse(card) {
      const lines = (card.innerText || '').split('\n').map((s) => s.trim()).filter(Boolean);
      // "4 | Worth the money | Review for: ... | <body> | <author> | , <city> | ... | · Jul, 2022"
      const authorIdx = lines.findIndex((l) => /^,\s*\S/.test(l));
      const bodyEnd = authorIdx > 0 ? authorIdx - 1 : Math.min(lines.length, 7);
      const body = lines.slice(2, bodyEnd)
        .filter((l) => !/^Review for:/.test(l) && !/^Helpful/.test(l))
        .join(' ');
      return {
        text: body,
        rating: /^[1-5](\.\d)?$/.test(lines[0]) ? lines[0] : '',
        date: (lines.find((l) => /[A-Z][a-z]{2},?\s*\d{4}/.test(l)) || '').replace(/^·\s*/, ''),
        reviewer: authorIdx > 0 ? lines[authorIdx - 1] : '',
        images: [...card.querySelectorAll('img[src*="rukminim"]')].map((i) => i.src),
      };
    },

    title: 'h1, span.VU-ZEz, span.B_NuCI',
    listingImages: 'img[src*="rukminim"]',
    productId: () => new URLSearchParams(location.search).get('pid'),
    // rukminim URLs embed the size: /image/480/640/... -> ask for the big one
    fullSize: (src) => src.replace(/\/image\/\d+\/\d+\//, '/image/832/832/'),
  },

  meesho: {
    // Meesho, like Flipkart, ships generated class names, so the card is found structurally.
    // Its stable marker is the posting line. CHECKED live 21 Sep 2026: 2 review cards parsed
    // with reviewer, rating, date, text and photos.
    // Block shape: "Nitin | 5.0 | Posted on 14 Aug 2026 | <text> | Helpful (7)"
    MARK: /Posted on\s+\d/,

    findCards() {
      const M = SITES.meesho.MARK;
      const seeds = [...document.querySelectorAll('div, span, p')].filter(
        (e) => M.test(e.innerText || '') && ![...e.children].some((c) => M.test(c.innerText || ''))
      );
      const cards = [];
      for (const s of seeds) {
        let p = s;
        for (let i = 0; i < 8 && p.parentElement; i++) {
          p = p.parentElement;
          const t = p.innerText || '';
          if (t.length > 60 && /^\s*\S[\s\S]*?\n\s*[1-5](\.\d)?\s*\n/.test(t)) break;
        }
        if (p && !cards.includes(p)) cards.push(p);
      }
      return cards;
    },

    parse(card) {
      const L = (card.innerText || '').split('\n').map((s) => s.trim()).filter(Boolean);
      const di = L.findIndex((l) => SITES.meesho.MARK.test(l));
      return {
        reviewer: L[0] || '',
        rating: L.find((l) => /^[1-5](\.\d)?$/.test(l)) || '',
        date: (L[di] || '').replace(/^Posted on\s*/, ''),
        text: L.slice(di + 1).filter((l) => !/^Helpful/i.test(l)).join(' '),
        // Meesho lazy-loads review photos: until the card is scrolled into view the <img>
        // has an empty src (AvifImage wrapper), so currentSrc and srcset are checked too.
        // If the reviews have not been scrolled past, there is genuinely no URL yet and the
        // review is judged on text alone -- which the card says.
        images: [...card.querySelectorAll('img')]
          .map((i) => i.currentSrc || i.getAttribute('src') || i.getAttribute('data-src')
            || (i.getAttribute('srcset') || '').split(' ')[0])
          .filter((u) => u && !u.startsWith('data:')),
      };
    },

    title: 'h1, span[class*="Title"]',
    listingImages: 'img[src*="images.meesho"]',
    productId: () => (location.pathname.match(/\/p\/([A-Za-z0-9]+)/) || [])[1],
    // Meesho serves .../original/xyz.jpg and .../312/312/xyz.jpg -- ask for the original
    fullSize: (src) => src.replace(/\/\d+\/\d+\//, '/original/'),
  },

  demo: {
    // the backend's /demo page reuses Amazon's markup
    review: '[data-hook="review"]',
    text: '[data-hook="review-body"]',
    rating: '[data-hook="review-star-rating"]',
    date: '[data-hook="review-date"]',
    author: '.a-profile-name',
    images: 'img[data-hook="review-image-tile"]',
    title: '#productTitle',
    listingImages: '#landingImage',
    productId: () => null,
    fullSize: (src) => src,
  },
};

function currentSite() {
  const h = location.hostname;
  if (h.includes('amazon.')) return SITES.amazon;
  if (h.includes('flipkart.')) return SITES.flipkart;
  if (h.includes('meesho.')) return SITES.meesho;
  if ((h === '127.0.0.1' || h === 'localhost') && location.pathname.startsWith('/demo')) return SITES.demo;
  return null;
}
