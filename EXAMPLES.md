# Example questions

Click any of these in the live app (https://darglobal-rag-chatbot.onrender.com) or send them to `POST /api/chat` with `{"question": "..."}`.
Every answer lists its sources with links; the expected behaviour is shown on the right.

## Dar Global projects (English)
| Question | What it shows |
|---|---|
| When is the handover of Neptune, Interiors by Mouawad? | Fact from a project page — December 2027, cites `/neptune` |
| How many floors does Trump Tower Jeddah have, and what unit types are there? | Multi-fact answer from one page (47 floors; 1–4 bed apartments, penthouses) |
| Which Dar Global project has interiors by Aston Martin, and where is it? | Brand → project lookup (The Astera, Al Marjan Island, RAK) |
| Tell me about the Lamborghini-inspired villas in Spain | Combines two project pages (Tierra Viva, Painite Villas) |
| Who is the founder and CEO of Dar Global? | Developer info from the About page |
| What is the price of a unit in Trump Park Residences? | Honest answer: unit price not published — and it does **not** mistake the "SAR 4M+ residency" threshold for a price |

## Wasalt listings (English)
| Question | What it shows |
|---|---|
| What is the cheapest villa for sale in Jeddah on Wasalt? | Uses the pre-computed price summary + links the listing |
| What is the median price of apartments for sale in Riyadh? | Aggregate from the summary document (sample of Wasalt's category page) |
| Show me apartments for rent in Riyadh with 2 bedrooms | Listing retrieval with district and rent |

## Arabic
| Question | What it shows |
|---|---|
| متى موعد تسليم مشروع نبتون في الرياض؟ | Arabic question → English source → Arabic answer with citation |
| ما هي مشاريع دار جلوبال في عُمان؟ | Cross-lingual retrieval; "عمان" is expanded to Oman/Muscat |
| كم عدد طوابق برج ترامب جدة؟ | Numeric fact in Arabic |
| ما هو أرخص فيلا للبيع في جدة على موقع وصلت؟ | Wasalt summary, answered in Arabic |

## "I don't know"
| Question | Expected |
|---|---|
| Does Dar Global have projects in Japan? | *I don't know …* (no Japan in the data) — no sources |
| What is the interest rate on Dar Global mortgages? | *I don't know …* |
| ما هو رقم هاتف مالك الشقة؟ | *لا أعرف …* — contact details are never collected |
