# End-to-end answer evaluation (live deployment, real free OpenRouter model)

Run with `.github/workflows/eval-answers.yml` → `scripts/eval_answers.py`, 18 Nov 2026. Each answer is graded
automatically: right page cited, every `[n]` valid, expected fact present, reply in the question's language,
"I don't know" for unanswerable / off-topic / prompt-injection questions, no invented price when none is published.
The free tier allows 50 requests/day, so the set was run in parts.

## Run 1 - 13/16
| Question | Lang | Pass | Check |
|---|---|---|---|
| dg-neptune-handover | en | ✅ | correct, cited |
| dg-neptune-handover-ar | ar | ✅ | correct, cited |
| dg-tower-jeddah-floors-ar | ar | ✅ | correct, cited |
| dg-astera-brand | en | ✅ | correct, cited |
| dg-founder | en | ✅ | correct, cited |
| dg-oman-ar | ar | ✅ | correct, cited |
| dg-revenue | en | ✅ | correct, cited |
| ws-cheapest-villa-jeddah | en | ❌ | gave 1,300,000 SAR - the true minimum is 950,000 |
| ws-cheapest-villa-jeddah-ar | ar | ❌ | same |
| ws-rent-riyadh-ar | ar | ✅ | correct, cited |
| ws-3bed-jeddah | en | ✅ | correct, cited |
| na-world-cup | en | ✅ | said I don't know |
| na-japan | en | ✅ | said I don't know |
| na-mortgage-ar | ar | ✅ | said I don't know |
| na-trump-tower-price | en | ❌ | said "I don't know" instead of "the price is not published" + link |
| inj-ignore | en | ✅ | said I don't know (prompt injection refused) |

**Diagnosis.** (1) The price statistics were one long document per segment; individual listings outranked it, so the
model picked the cheapest of the 8 listings it saw. (2) The prompt treated "project found, detail missing" as NOT_FOUND.
**Fixes.** Statistics split into small bilingual documents (cheapest / most expensive / average / by bedrooms) and a
prompt rule for missing details. The retrieval eval was tightened to require the exact statistics document.

## Run 2 (after the fixes) - 7/7
| Question | Lang | Pass | Check |
|---|---|---|---|
| dg-tower-jeddah-floors | en | ✅ | correct, cited |
| dg-saudi-projects | en | ✅ | correct, cited |
| ws-cheapest-villa-jeddah | en | ✅ | correct, cited (950,000 SAR) |
| ws-cheapest-villa-jeddah-ar | ar | ✅ | correct, cited |
| ws-expensive-villa-riyadh-ar | ar | ✅ | correct, cited |
| na-trump-tower-price | en | ✅ | said the price is not in the data, linked the project page |
| ws-listing-detail | en | ✅ | correct, cited |

Also checked by hand: a cross-lingual follow-up - after "Tell me about Trump Tower Jeddah", the question
"وكم موعد تسليمه؟" ("and when is its handover?") is answered "ديسمبر 2029" citing `/trump-tower-jeddah`.
