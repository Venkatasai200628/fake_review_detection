"""
gen_text2.py -- v6.2 review text: same styles as gen_text.py, far larger space.

WHY THIS EXISTS
gen_text.py writes each review by picking ONE finished sentence out of a small list
(DECEPTIVE_POSITIVE has 8 fixed strings, CAMPAIGN_TEMPLATES 5, LAZY_GENUINE 8). With
3,348 rows to fill, the same sentence has to be used over and over, and it lands in the
training half and the test half alike. Measured in out/model_comparison/TEXT_DIFFICULTY_AUDIT.md:

    46.2% of test reviews are word-for-word copies of a training review
    (text_deception 69%, coordinated_reuse 100%)

so "copy the label of the most similar training review" -- a rule with no learning in it
at all -- already scores 100% on fake text. Every text model then looks perfect on that
group, ours included, and the number means nothing.

THE FIX
Each review is ASSEMBLED from independent slots instead of chosen whole, so the space is
tens of thousands of sentences per style rather than a handful. The styles, and the
deliberate overlap between them, are unchanged -- that overlap is the point of the
project (guide section: OVERLAP CASES):

  genuine detailed        concrete part, situation, duration, specific failure
  lazy genuine            short, generic, positive -- a real customer, low effort
  deceptive generic       superlatives, no verifiable detail, exclamation marks
  sophisticated deceptive fabricated specifics, written to look like genuine detail
  campaign                shared talking points; members of ONE campaign stay
                          near-identical to each other, but every campaign differs

Nothing here reads a label. Ratings are not produced here: regen_text.py conditions the
text on the rating the row already has, so the existing rating/sentiment agreement (and
the v6.1 removal of the rating shortcut) is preserved exactly.
"""
import random

from gen_text import CATEGORY_VOCAB, CATEGORY_WORD, DURATIONS, GENERIC_FALLBACK  # noqa: F401

# --------------------------------------------------------------- genuine, detailed
G_OPEN = [
    "Been using this {sit} for about {dur} now",
    "Bought it {sit} and have had it {dur}",
    "Using it {sit} for {dur}",
    "Had this {dur}, mainly {sit}",
    "Picked one up {sit} roughly {dur} ago",
    "Ordered this {sit}, {dur} of use so far",
    "This has been in use {sit} for {dur}",
    "Got it {sit} and it has seen {dur} of use",
    "Around {dur} of using this {sit}",
    "I have had it {dur}, used mostly {sit}",
    "Bought this {sit}, been about {dur}",
    "Roughly {dur} in, using it {sit}",
]
G_POS = [
    "and the {part} has held up well",
    "and the {part} is still solid",
    "and the {part} shows no real wear",
    "and honestly the {part} is the best bit",
    "and the {part} is better than I expected for the price",
    "and the {part} has given me no trouble",
    "and the {part} still feels new",
    "and the {part} does exactly what it needs to",
    "and the {part} has been the strong point",
    "and nothing has gone wrong with the {part}",
]
G_NEG = [
    "and then {fail}",
    "and {fail}, which was annoying",
    "before {fail}",
    "and unfortunately {fail}",
    "and eventually {fail}",
    "and then {fail}, so that was that",
    "and after that {fail}",
    "and sure enough {fail}",
]
G_MIX = [
    "and while the {part} is fine, {fail}",
    "and the {part} is good, but {fail}",
    "and the {part} holds up, although {fail}",
    "and it is decent, though {fail}",
    "and the {part} is solid, even if {fail}",
]
G_CLOSE_POS = [
    "No complaints so far.", "Happy with the purchase.", "Would buy it again.",
    "Does the job properly.", "Nothing fancy, but it works.", "Pleased with it overall.",
    "Worth what I paid.", "It has earned its place.", "Simple and it works.",
    "I would get another.", "Fine for what I needed.", "",
]
G_CLOSE_NEG = [
    "Disappointing for what it cost.", "I would not buy again.",
    "Not built for regular use.", "Expected more at this price.",
    "Back it goes.", "Would not recommend it.", "A shame, it started well.",
    "Not worth the money.", "Looking for a replacement now.", "",
]
G_CLOSE_MIX = [
    "Fine if you use it lightly.", "Acceptable for the price.",
    "Depends what you need it for.", "Not perfect, but usable.",
    "It is a compromise.", "Middle of the road really.", "",
]

# --------------------------------------------------------------- lazy genuine
L_A = [
    "Good product,", "Nice quality,", "Very good,", "Works well,", "Decent product,",
    "Solid buy,", "Pretty good,", "Happy with it,", "All good,", "Does the job,",
    "Nice little thing,", "Not bad at all,", "Quite good,", "Fine product,",
    "Works a treat,", "Really nice,",
]
L_B = [
    "works fine.", "does the job.", "no issues so far.", "exactly what I wanted.",
    "no complaints.", "as expected.", "worth the price.", "arrived on time.",
    "good value.", "easy to use.", "seems well made.", "nothing to complain about.",
    "came quickly.", "does what it should.",
]
L_C = [
    "Happy with it.", "Recommend.", "Thanks!", "Good value.", "Would buy again.",
    "Satisfied.", "No complaints at all.", "Pleased.", "Cheers!", "Does what it says.",
    "Glad I got it.", "",
]

# --------------------------------------------------------------- deceptive, generic
D_OPEN = [
    "Amazing product!", "Best purchase ever.", "Outstanding quality!", "Absolutely love it!",
    "Superb!", "Perfect product.", "Wonderful item!", "Excellent!", "Fantastic buy!",
    "Brilliant!", "Top notch!", "So happy with this!", "Really impressed!", "Five stars!",
    "Incredible value!", "Just perfect!", "Simply the best!", "Great stuff!",
]
D_QUAL = [
    "Excellent quality and superb value.", "The quality is top class.",
    "Great quality for the money.", "Premium feel and finish.", "Quality is unmatched.",
    "Build quality is amazing.", "Superb quality throughout.", "Really good quality item.",
    "Quality exceeded expectations.", "First class quality.", "The finishing is superb.",
    "Very good quality product.", "Quality is simply the best.", "Great value and quality.",
    "Worth every penny.", "Money well spent.",
]
D_EXTRA = [
    "Fast delivery too.", "Packaging was perfect.", "Arrived earlier than expected.",
    "Seller was excellent.", "Shipping was very quick.", "Exactly as described.",
    "Exceeded all my expectations.", "No issues whatsoever.", "Nothing to complain about.",
    "Everything was perfect.", "Great service from the seller.", "Delivery was super fast.", "",
]
D_CTA = [
    "Highly recommend to everyone!", "Would definitely recommend!", "Buy it now!",
    "Will buy again!", "Highly satisfied!", "Recommended to all!", "You won't regret it!",
    "Definitely worth buying!", "Recommend without hesitation!", "Five stars from me!",
    "Go for it!", "Don't think twice!",
]
DN_OPEN = [
    "Terrible product.", "Worst purchase.", "Awful.", "Complete rubbish.",
    "Very disappointed.", "Do not buy.", "Total letdown.", "Shocking quality.",
    "Waste of money.", "Absolutely terrible.",
]
DN_BODY = [
    "Complete waste of money.", "Poor quality item.", "Cheap rubbish, nothing like the description.",
    "Very bad quality throughout.", "Nothing like what was advertised.", "Fell apart immediately.",
    "Cheaply made and useless.", "Quality is shocking.", "Not worth a single star.",
    "Badly made and overpriced.",
]
DN_CTA = [
    "Do not buy this.", "Not recommended at all.", "Avoid this seller completely.",
    "Stay well away.", "Save your money.", "Would give zero stars.",
    "Totally disappointed.", "Never again.",
]

# --------------------------------------------------------------- sophisticated deceptive
S_OPEN = [
    "Been using this {sit} for {dur}",
    "Picked this up {sit} about {dur} ago",
    "Using it {sit} for {dur} now",
    "Had it {dur}, mostly {sit}",
    "Bought it {sit} and {dur} later",
    "About {dur} of use {sit}",
    "Ordered it {sit}, {dur} ago now",
    "Running it {sit} for {dur}",
]
S_BODY = [
    "and the {part} still looks new",
    "and the {part} shows no wear at all",
    "and the {part} is well made, everything still works",
    "and the {part} has held up better than my last one",
    "and the {part} is flawless",
    "and the {part} has not given me a moment of trouble",
    "and the {part} is as good as day one",
    "and the {part} still performs perfectly",
]
S_CLOSE = [
    "Genuinely impressed with the build.", "Exactly what I hoped for.",
    "Very pleased with this one.", "Better than I expected honestly.",
    "Really can't fault it.", "Would happily buy another.",
    "Quality you can feel.", "No regrets at all.",
    "Does everything it promises.", "Glad I went with this one.",
]

# --------------------------------------------------------------- campaign
C_OPEN = [
    "Really happy with this {cat_word}.", "Very happy with this {cat_word}.",
    "Great {cat_word}.", "Excellent {cat_word}.", "Lovely {cat_word}.",
    "Superb {cat_word}.", "Very nice {cat_word}.", "Brilliant {cat_word}.",
    "Pleased with this {cat_word}.", "Top {cat_word}.", "Fantastic {cat_word}.",
    "Delighted with this {cat_word}.",
]
C_QUAL = [
    "The quality is great", "Quality is excellent", "The quality is really good",
    "Build quality is great", "The finish is excellent", "Quality feels premium",
    "The quality is spot on", "Really well made", "Quality is first rate",
    "The materials feel good", "Solidly made", "The quality is superb",
]
C_DELIV = [
    "and it arrived quickly.", "and delivery was fast.", "and it came very quickly.",
    "and it arrived ahead of time.", "and shipping was quick.", "and it turned up fast.",
    "and postage was speedy.", "and it arrived next day.", "and delivery was prompt.",
    "and it got here fast.",
]
C_CTA = [
    "Recommended.", "Would recommend.", "Happy to recommend.", "Recommend it to anyone.",
    "Definitely recommend.", "Would buy again.", "Very pleased.", "No complaints.",
    "Five stars.", "Great buy.",
]


def _vocab(category):
    return CATEGORY_VOCAB.get(category, GENERIC_FALLBACK)


def _join(bits):
    return ' '.join(b for b in bits if b).replace(' ,', ',').strip()


def _stitch(opener, body, rng):
    """Join the two halves as one clause or as two sentences -- more variety, fewer
    'and ... and ...' runs when the opener already contains 'and'."""
    if rng.random() < 0.5 and body.startswith('and '):
        rest = body[4:]
        return opener + '. ' + rest[0].upper() + rest[1:]
    return opener + ' ' + body


def genuine_detailed(category, rng, sentiment):
    parts, sits, fails = _vocab(category)
    slot = dict(part=rng.choice(parts), sit=rng.choice(sits),
                fail=rng.choice(fails), dur=rng.choice(DURATIONS))
    body, close = {'pos': (G_POS, G_CLOSE_POS), 'neg': (G_NEG, G_CLOSE_NEG),
                   'mixed': (G_MIX, G_CLOSE_MIX)}[sentiment]
    first = _stitch(rng.choice(G_OPEN), rng.choice(body), rng).format(**slot)
    return _join([first + '.', rng.choice(close)])


def lazy_genuine(rng):
    return _join([rng.choice(L_A), rng.choice(L_B), rng.choice(L_C)])


def deceptive_generic(rng, polarity):
    if polarity == 'neg':
        return _join([rng.choice(DN_OPEN), rng.choice(DN_BODY), rng.choice(DN_CTA)])
    return _join([rng.choice(D_OPEN), rng.choice(D_QUAL), rng.choice(D_EXTRA), rng.choice(D_CTA)])


def sophisticated_deceptive(category, rng):
    parts, sits, fails = _vocab(category)
    slot = dict(part=rng.choice(parts), sit=rng.choice(sits), dur=rng.choice(DURATIONS))
    first = _stitch(rng.choice(S_OPEN), rng.choice(S_BODY), rng).format(**slot)
    return _join([first + '.', rng.choice(S_CLOSE)])


def campaign_text(category, campaign_seed, rng):
    """
    One campaign = one set of talking points. Members stay near-identical to each other
    (that is the coordination signal the graph features are meant to find), while
    different campaigns read differently.
    """
    local = random.Random(campaign_seed)
    word = CATEGORY_WORD.get(category, 'product')
    opens = local.sample(C_OPEN, 2)
    quals = local.sample(C_QUAL, 2)
    deliv = local.sample(C_DELIV, 2)
    ctas = local.sample(C_CTA, 2)
    return _join([rng.choice(opens).format(cat_word=word),
                  rng.choice(quals) + ' ' + rng.choice(deliv),
                  rng.choice(ctas)])


def space_size():
    """Rough count of distinct sentences each style can produce (for the audit)."""
    return {
        'genuine detailed (per category, per sentiment)':
            2 * len(G_OPEN) * len(G_POS) * len(G_CLOSE_POS) * 5 * 4 * len(DURATIONS),
        'lazy genuine': len(L_A) * len(L_B) * len(L_C),
        'deceptive generic positive': len(D_OPEN) * len(D_QUAL) * len(D_EXTRA) * len(D_CTA),
        'deceptive generic negative': len(DN_OPEN) * len(DN_BODY) * len(DN_CTA),
        'sophisticated deceptive (per category)':
            2 * len(S_OPEN) * len(S_BODY) * len(S_CLOSE) * 5 * 4 * len(DURATIONS),
        'campaign (across campaigns)': len(C_OPEN) * len(C_QUAL) * len(C_DELIV) * len(C_CTA),
        'campaign (within one campaign)': 2 * 2 * 2 * 2,
    }
