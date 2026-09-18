"""
gen_text.py — review text generated CONDITIONED ON the image's fraud_type.

Text is never independent of the image (PROJECT_SPEC.md 4.3):

  genuine              -> concrete specifics: a named part, a duration, a real
                          usage situation
  visual_manipulation  -> text sounds authentic on purpose; the IMAGE is the
                          only tell
  text_deception       -> generic superlatives, no verifiable detail; the TEXT
                          is the only tell
  coordinated_reuse    -> near-duplicate phrasing from a small shared template
                          pool, simulating recruited reviewers given talking
                          points

The specificity gap between the first and third case is the documented spam
signal this module reproduces.
"""

# Per-category vocabulary: (parts, positive situations, failure modes)
CATEGORY_VOCAB = {
    'Bluetooth speaker': (
        ['bass driver', 'charging port', 'pairing button', 'rubber seal', 'battery'],
        ['on the balcony during a barbecue', 'in the shower', 'at a small house party',
         'on a camping trip', 'while cooking dinner'],
        ['started crackling above half volume', 'stopped holding a charge',
         'lost pairing every few minutes', 'developed a rattle in the bass']),
    'Handbag': (
        ['shoulder strap', 'zip pull', 'inner lining', 'magnetic clasp', 'base studs'],
        ['for daily commuting', 'on a weekend trip', 'as a work bag',
         'for carrying a laptop and a lunch box'],
        ['the stitching came loose at the strap', 'the lining tore',
         'the zip started catching', 'the clasp stopped holding shut']),
    'Knifes': (
        ['blade edge', 'handle rivets', 'bolster', 'tang', 'grip'],
        ['for daily vegetable prep', 'when breaking down a chicken',
         'for slicing bread', 'in a busy kitchen'],
        ['the edge dulled within a month', 'the handle worked loose',
         'a small chip appeared near the tip', 'it started rusting at the bolster']),
    'Office cair': (
        ['lumbar support', 'gas lift', 'armrest', 'castor wheels', 'seat foam'],
        ['for eight hour work days', 'while studying late', 'in a home office',
         'for long gaming sessions'],
        ['the gas lift sank after a few weeks', 'the seat foam flattened',
         'an armrest developed play', 'a castor cracked']),
    'Phone case': (
        ['corner bumpers', 'camera lip', 'button covers', 'inner lining', 'port cutout'],
        ['after a waist height drop onto tile', 'for daily pocket carry',
         'with a screen protector fitted', 'on a work site'],
        ['the corners yellowed', 'the button covers went mushy',
         'it stretched and stopped gripping', 'the camera lip wore down']),
    'Shoes': (
        ['outsole grip', 'heel counter', 'insole', 'toe box', 'laces'],
        ['for a daily walk to work', 'on a long airport day', 'for gym sessions',
         'on wet pavement'],
        ['the sole started separating at the toe', 'the insole compressed flat',
         'the heel lining wore through', 'the grip went on wet floors']),
    'Sunglasses': (
        ['hinge screws', 'nose pads', 'lens coating', 'temple arms', 'frame'],
        ['for driving', 'on a beach holiday', 'for daily commuting',
         'while cycling'],
        ['a hinge screw worked loose', 'the coating started flaking',
         'the nose pads discoloured', 'the arms lost their tension']),
    'Watches': (
        ['clasp', 'crown', 'strap pins', 'crystal', 'bezel'],
        ['as a daily watch', 'for a formal event', 'while swimming',
         'at the gym'],
        ['the clasp stopped locking properly', 'it gained a few minutes a week',
         'the crystal picked up a scratch', 'a strap pin sheared']),
    'Water bottle': (
        ['lid gasket', 'threading', 'carry loop', 'inner coating', 'spout'],
        ['for daily gym use', 'on hikes', 'for keeping tea hot at work',
         'in a car cup holder'],
        ['it started leaking at the lid', 'the coating flaked inside',
         'it stopped holding temperature', 'the threading cross-threads easily']),
    'Wireless_Earbuds': (
        ['ear tips', 'charging case hinge', 'touch controls', 'battery', 'mic'],
        ['on daily commutes', 'during workouts', 'for work calls',
         'on a long flight'],
        ['the left bud stopped charging', 'the case hinge went loose',
         'battery life dropped noticeably', 'the touch controls got erratic']),
    'Yoga mat': (
        ['surface texture', 'edge binding', 'foam density', 'underside grip'],
        ['for daily morning practice', 'in a heated class', 'for floor workouts',
         'on a hardwood floor'],
        ['it started flaking at the edges', 'the grip went once it got sweaty',
         'a permanent crease formed', 'the foam compressed under the knees']),
    'backpack': (
        ['shoulder straps', 'laptop sleeve', 'main zip', 'base fabric', 'chest clip'],
        ['for daily college carry', 'on a week of travel', 'for carrying gym kit',
         'in heavy rain'],
        ['a zip tooth broke', 'the strap padding compressed',
         'the base fabric started fraying', 'it soaked through in rain']),
    'mouse': (
        ['left click switch', 'scroll wheel', 'side buttons', 'sensor', 'glide feet'],
        ['for daily office work', 'for gaming', 'with a laptop while travelling',
         'on a glass desk'],
        ['the left click started double-clicking', 'the scroll wheel got notchy',
         'a side button stopped registering', 'the glide feet wore down']),
    'skincare set': (
        ['pump dispenser', 'jar seal', 'texture', 'fragrance', 'consistency'],
        ['as a nightly routine', 'through a dry winter', 'on sensitive skin',
         'alongside an existing routine'],
        ['the pump stopped drawing product', 'the texture separated',
         'it broke me out after a fortnight', 'the seal leaked in a bag']),
    'suitcases': (
        ['spinner wheels', 'telescopic handle', 'zip track', 'shell corners', 'latches'],
        ['on a two week trip', 'through three connecting flights',
         'as a check-in bag', 'for monthly work travel'],
        ['a wheel seized', 'the handle started sticking halfway',
         'a shell corner cracked', 'the zip split under load']),
}

# v6: the image-pool folder was renamed Handbag -> Handbags
CATEGORY_VOCAB['Handbags'] = CATEGORY_VOCAB['Handbag']

GENERIC_FALLBACK = (
    ['build quality', 'finish', 'packaging', 'materials'],
    ['in daily use', 'over the past month', 'for regular use'],
    ['it wore out quickly', 'the quality dropped off', 'it stopped working properly'])


# --- genuine: concrete, specific, varied sentiment -------------------------

GENUINE_POSITIVE = [
    "Been using this {sit} for about {dur} now and the {part} has held up well. No complaints so far.",
    "Bought it {sit}. The {part} is better than I expected for the price, still solid after {dur}.",
    "Using it {sit} for {dur}. The {part} does exactly what it needs to, nothing fancy but it works.",
    "Had this {dur}. Mainly use it {sit}. The {part} is the part that impressed me most.",
]

GENUINE_NEGATIVE = [
    "Used it {sit} for {dur} and then {fail}. Disappointing for what it cost.",
    "Worked fine at first but after {dur} {fail}. The {part} was the weak point.",
    "Bought it {sit}. Within {dur} {fail}, so I would not buy again.",
    "About {dur} in and {fail}. The {part} clearly was not built for regular use.",
]

GENUINE_MIXED = [
    "Mixed feelings. The {part} is good but after {dur} {fail}. Fine if you use it lightly.",
    "Decent {sit}, though {fail} after {dur}. The {part} is still holding up at least.",
    "Does the job {sit}. Not perfect, {fail} eventually, but for the price it is acceptable.",
]

DURATIONS = ['two weeks', 'a month', 'six weeks', 'three months', 'four months',
             'half a year', 'about eight months']


# --- text_deception: generic superlatives, zero verifiable detail ----------

DECEPTIVE_POSITIVE = [
    "Amazing product! Excellent quality and superb value. Highly recommend to everyone!",
    "Best purchase ever. Perfect in every way. Five stars, will buy again!",
    "Outstanding quality! Exceeded all my expectations. Fast delivery too. Recommended!",
    "Absolutely love it! Great product, great price, great seller. Buy it now!",
    "Superb! Fantastic quality and amazing value for money. Very happy with this purchase!",
    "Perfect product, exactly as described. Excellent quality. Highly satisfied!",
    "Wonderful item! Top quality and great service. Would definitely recommend!",
    "Excellent! Very good product with superb finishing. Worth every penny!",
]

DECEPTIVE_NEGATIVE = [
    "Terrible product. Complete waste of money. Do not buy this. Very bad quality.",
    "Worst purchase. Poor quality item. Totally disappointed. Not recommended at all.",
    "Awful. Cheap rubbish, nothing like the description. Avoid this seller completely.",
]


# --- coordinated_reuse: shared talking points, near-duplicate phrasing -----

CAMPAIGN_TEMPLATES = [
    "Really happy with this {cat_word}. The quality is great and it arrived quickly. Recommended.",
    "This {cat_word} is great quality and arrived fast. Very happy with it, recommended.",
    "Very happy with this {cat_word}. Great quality, quick delivery. Would recommend.",
    "Great {cat_word}, quality is excellent and delivery was quick. Happy to recommend.",
    "Excellent {cat_word}. The quality is really good and it came fast. Recommended.",
]

CATEGORY_WORD = {
    'Bluetooth speaker': 'speaker', 'Handbag': 'bag', 'Handbags': 'bag', 'Knifes': 'knife set',
    'Office cair': 'chair', 'Phone case': 'case', 'Shoes': 'pair of shoes',
    'Sunglasses': 'pair of sunglasses', 'Watches': 'watch',
    'Water bottle': 'bottle', 'Wireless_Earbuds': 'set of earbuds',
    'Yoga mat': 'mat', 'backpack': 'backpack', 'mouse': 'mouse',
    'skincare set': 'set', 'suitcases': 'suitcase',
}


# ---------------------------------------------------------------------------
# OVERLAP CASES -- added in v3.
#
# v2 text was trivially separable: a text-only baseline reached PR-AUC 1.000,
# which made every ablation meaningless. Real review text does not separate that
# cleanly, for two reasons that are now modelled explicitly:
#   1. plenty of GENUINE customers write short, generic, low-effort reviews
#   2. skilled paid reviewers fabricate specific-sounding detail on purpose
# These two groups overlap in text space, so text alone cannot resolve them and
# the image/behaviour evidence has to do the work. That is the whole premise of
# the project, and the dataset must contain it.
# ---------------------------------------------------------------------------

LAZY_GENUINE = [
    "Good product, works fine. Happy with it.",
    "Nice quality, does the job. Recommend.",
    "Very good, exactly what I wanted. Thanks!",
    "Works well, no issues so far. Good value.",
    "Decent product for the price. Satisfied.",
    "Great, arrived on time and works well.",
    "Solid buy, no complaints at all.",
    "Pretty good overall, would buy again.",
]

# Fabricated specifics -- structurally identical to genuine text, so surface
# features cannot tell them apart.
SOPHISTICATED_DECEPTIVE = [
    "Been using this {sit} for {dur} and the {part} still looks new. Genuinely impressed with the build.",
    "Picked this up {sit}. After {dur} the {part} shows no wear at all. Exactly what I hoped for.",
    "Using it {sit} for {dur} now. The {part} is well made and everything still works perfectly.",
    "Had it {dur} {sit}. The {part} has held up better than the last one I owned. Very pleased.",
]


def gen_lazy_genuine_text(rng):
    """A real customer writing a short low-effort review."""
    return rng.choice(LAZY_GENUINE)


def gen_sophisticated_deceptive_text(category, rng):
    """A paid reviewer fabricating convincing detail."""
    parts, sits, fails = _vocab(category)
    tpl = rng.choice(SOPHISTICATED_DECEPTIVE)
    return tpl.format(part=rng.choice(parts), sit=rng.choice(sits),
                      dur=rng.choice(DURATIONS))


def _vocab(category):
    return CATEGORY_VOCAB.get(category, GENERIC_FALLBACK)


def gen_genuine_text(category, rng, sentiment=None):
    parts, sits, fails = _vocab(category)
    sentiment = sentiment or rng.choices(
        ['pos', 'neg', 'mixed'], weights=[0.62, 0.22, 0.16])[0]
    pool = {'pos': GENUINE_POSITIVE, 'neg': GENUINE_NEGATIVE,
            'mixed': GENUINE_MIXED}[sentiment]
    tpl = rng.choice(pool)
    text = tpl.format(part=rng.choice(parts), sit=rng.choice(sits),
                      fail=rng.choice(fails), dur=rng.choice(DURATIONS))
    return text, sentiment


def gen_deceptive_text(rng):
    """
    Returns (text, polarity). Polarity is returned so the rating stays CONSISTENT
    with the text. An unintended rating/text contradiction would hand the model a
    shortcut signal that is not part of the stated design.
    """
    if rng.random() < 0.85:
        return rng.choice(DECEPTIVE_POSITIVE), 'pos'
    return rng.choice(DECEPTIVE_NEGATIVE), 'neg'


def gen_campaign_rating(rng):
    """Campaign text is always positive, so the rating must be too."""
    return 5 if rng.random() < 0.92 else 4


def gen_campaign_text(category, rng, campaign_seed):
    """
    All members of one campaign draw from the same 2-3 templates, so the texts
    are near-duplicates of each other but not byte-identical.
    """
    local = __import__('random').Random(campaign_seed)
    chosen = local.sample(CAMPAIGN_TEMPLATES, 3)
    tpl = rng.choice(chosen)
    word = CATEGORY_WORD.get(category, 'product')
    return tpl.format(cat_word=word)


def gen_rating(fraud_type, rng, sentiment=None):
    """
    Rating skew: fraud clusters at 1 or 5, rarely 2-3 (incentivized-review
    literature). Genuine ratings follow a realistic positive-leaning spread and
    stay consistent with the text sentiment.
    """
    if fraud_type in ('text_deception', 'coordinated_reuse', 'visual_manipulation'):
        # Fraud ratings cluster at the extremes, but must still AGREE with the
        # polarity of the generated text (see gen_deceptive_text docstring).
        if sentiment == 'neg':
            return 1 if rng.random() < 0.85 else 2
        return 5 if rng.random() < 0.92 else 4
    if sentiment == 'pos':
        return rng.choices([4, 5], weights=[0.35, 0.65])[0]
    if sentiment == 'neg':
        return rng.choices([1, 2], weights=[0.55, 0.45])[0]
    return rng.choices([2, 3, 4], weights=[0.25, 0.45, 0.30])[0]
