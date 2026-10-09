# Weekly-company experiments: retrieval comparison

Definitions, per experiment:
- keyword retrieval (K): articles naming at least one strong entity.
- semantic retrieval (S): articles scoring >= that experiment's semantic threshold.
- relevant: the judge's yes, over everything it saw (candidates, plus a recall audit of the
  15 highest-scoring non-candidates and every non-candidate matching an ambiguous name).
  Articles the judge never saw count as not relevant, so recall figures are relative to
  what was found, not absolute.

| | OpenAI This Week | Google AI This Week |
|---|---|---|
| Articles | 250 | 250 |
| Keyword hits (K) | 77 | 37 |
| Semantic hits (S) | 30 | 21 |
| Overlap K ∩ S | 26 | 15 |
| Candidates judged | 83 | 53 |
| Judged relevant (incl. audit finds) | 34 | 25 |
| Relevant found only semantically | 0 | 0 |
| Misses in recall audit | 0 | 3 |

## OpenAI This Week

250 articles · semantic threshold 0.5 · ambiguous-name threshold 0.4 · 34 judged relevant in total

| Retrieval route | Articles | Precision (judged relevant) | Recall (of all relevant found) |
|---|---|---|---|
| Keyword (K), all | 77 | 34/77 (44%) | 34/34 (100%) |
| Semantic (S), all | 30 | 22/30 (73%) | 22/34 (65%) |
| Both K and S | 26 | 22/26 (85%) | 22/34 (65%) |
| Keyword only (K, not S) | 51 | 12/51 (24%) | 12/34 (35%) |
| Semantic only (S, not K) | 4 | 0/4 (0%) | 0/34 (0%) |
| Ambiguous name + similarity only | 2 | 0/2 (0%) | 0/34 (0%) |

**Similarity ranking** (precision among the top-k by score, all judged):

- top 10: 10/10 (100%)
- top 20: 18/20 (90%)
- top 30: 22/30 (73%)
- median score: relevant 0.528, judged not relevant 0.452; lowest relevant 0.362

**Recall audit:** 0 of 17 non-candidates judged relevant (the 15 highest-scoring, plus every other ambiguous-name match; scores 0.286–0.499).

**Relevant articles found only by semantic search** (0)

- none: every relevant article also matched a keyword

**Relevant articles keywords found but semantic search ranked below the threshold** (12)

- 0.362 · Hot Girl Hotline is like ‘Dear Abby’ for the AI era — _The article discusses ChatGPT users seeking relationship advice and reports lawsuits alleging its manipulative conversations caused mental-health harms._
- 0.409 · Lawyer Cites ChatGPT-Invented Fake Witnesses in Murder Appeal — _The article centers on ChatGPT generating fabricated witnesses and testimony in a lawyer’s court brief, highlighting hallucinations and their legal consequences._
- 0.414 · Brian Chesky interview: AI agents need their own operating system — _The article discusses ChatGPT’s attempted app store and quotes Brian Chesky telling Sam Altman how to build it, directly addressing an OpenAI product and leadership._
- 0.438 · Opus 5.5 loves to tell you ‘this matters’ (and other AI writing tells) — _The article analyzes OpenAI’s Astra writing quirks and cites its GPT-6 model releases and claims about improved clarity._
- 0.448 · OpenAI and Synopsys team up to build an AI model that designs chips like a seasoned engineer — _OpenAI is a main subject: it partnered with Synopsys to build and commercialize GPT-Synopsys, a specialized chip-design model running on OpenAI infrastructure._
- 0.453 · Businesses are using more AI and paying less for it, Ramp AI Index shows — _The article reports OpenAI’s role in driving AI price competition with Anthropic and its 44.5% share of business token spending._
- 0.459 · Google drops Gems for Skills, joining OpenAI and Anthropic in the shift to agent-ready prompt formats — _The article reports that OpenAI is sunsetting Custom GPTs while Google replaces Gems with Skills, positioning the change within agent-ready prompt formats._
- 0.462 · OpenAI agents tried to hack Wikipedia tools and flooded it with traffic — _The article reports that OpenAI agents attempted unauthorized edits and hacking of Wikimedia tools, generating heavy traffic and potentially disrupting Wikipedia infrastructure._
- 0.467 · OpenAI “rogue” agent activities found on Wikimedia projects — _The article reports unauthorized OpenAI agents editing Wikimedia wikis, attempting infrastructure exploits, and generating heavy traffic and data queries._
- 0.470 · The ugly economics of consumer AI — _The article analyzes OpenAI’s consumer AI economics, Dots launch, operating costs, and reportedly successful pivot toward enterprise bookings._
- 0.478 · llm-openai-decisions 0.1a0 — _The article reports OpenAI releasing a new Decisions API and describes its model, capabilities, pricing, and related plugin._
- 0.494 · OpenAI’s Jev clone could help the frontier lab stop its swarming agents — _The article centers on OpenAI’s newly announced Decisions API, its similarity to Jev, and OpenAI’s agent-security measures._

**False positives** (candidates the judge rejected): 49 — keyword only 39, semantic only 4, both 4, ambiguous name 2

**Semantic-only false positives** (4)

- 0.530 · The latest AI news we announced in September 2026 — _The article is Google's roundup of Gemini and other Google AI announcements; “Atlas” refers to Google’s AlphaGenome Atlas, not OpenAI._
- 0.525 · Can Safeworld convince people that GenAI robots won’t hurt them? — _The article focuses on Safeworld’s robot-safety startup; OpenAI is not mentioned and generative AI is discussed only generally._
- 0.513 · What AI gets wrong and what failure teaches us — _The article is about Microsoft researcher Jennifer Neville’s AI evaluation work; OpenAI is not mentioned or a subject of the discussion._
- 0.503 · Open or closed AI? How founders are choosing what to build on at TechCrunch Disrupt 2026 — _The article discusses startups’ choices between open and proprietary AI models generally, without mentioning OpenAI or its products, research, business, or people._

## Google AI This Week

250 articles · semantic threshold 0.5 · ambiguous-name threshold 0.45 · 25 judged relevant in total

| Retrieval route | Articles | Precision (judged relevant) | Recall (of all relevant found) |
|---|---|---|---|
| Keyword (K), all | 37 | 22/37 (59%) | 22/25 (88%) |
| Semantic (S), all | 21 | 11/21 (52%) | 11/25 (44%) |
| Both K and S | 15 | 11/15 (73%) | 11/25 (44%) |
| Keyword only (K, not S) | 22 | 11/22 (50%) | 11/25 (44%) |
| Semantic only (S, not K) | 6 | 0/6 (0%) | 0/25 (0%) |
| Ambiguous name + similarity only | 10 | 0/10 (0%) | 0/25 (0%) |

**Similarity ranking** (precision among the top-k by score, all judged):

- top 10: 9/10 (90%)
- top 20: 11/20 (55%)
- top 30: 11/30 (37%)
- median score: relevant 0.477, judged not relevant 0.474; lowest relevant 0.355

**Recall audit:** 3 of 30 non-candidates judged relevant (the 15 highest-scoring, plus every other ambiguous-name match; scores 0.304–0.499).

- MISSED 0.442 · Google froze its open source bug bounty program due to a ‘significant rise’ in AI submissions (Google x6) — _Google paused its open-source vulnerability rewards program because AI-generated automated submissions overwhelmed engineers and maintainers._
- MISSED 0.400 · Internet Infrastructure Services Empower Deepfake Abuse, New Study Finds (Google x9) — _The article examines Google’s provision of SSL certificates and advertising to deepfake-abuse sites, alongside its policies, removal process, and Search-ranking changes._
- MISSED 0.355 · MCP for agent-to-agent comms may be the riskiest protocol you've never heard of (Google x2) — _The article reports that Google acknowledged vulnerabilities in its AI-agent systems involving prompt injection and MCP trust gaps._

**Relevant articles found only by semantic search** (0)

- none: every relevant article also matched a keyword

**Relevant articles keywords found but semantic search ranked below the threshold** (11)

- 0.428 · With most information hidden, the game Stratego had stumped AI—until now — _The article discusses DeepMind’s DeepNash and its inability to reliably master Stratego, directly covering Google DeepMind’s AI research and limitations._
- 0.431 · Opus 5.5 loves to tell you ‘this matters’ (and other AI writing tells) — _The article analyzes Gemini 3.1 Pro’s AI-generated writing patterns, noting it has nearly eliminated em-dashes compared with human writing._
- 0.432 · Satlyt, founded by a former Google and SpaceX product manager, raises $8M to run AI on satellites — _Satlyt deployed Google DeepMind’s Gemma AI model aboard a spacecraft, demonstrating Google’s AI technology in an orbital computing application._
- 0.441 · AutoSynthData: Generating Training Data for Enterprise Agents — _The article evaluates and fine-tunes Google's Gemma-4 model using synthetic enterprise-agent training data, reporting substantial benchmark improvements._
- 0.445 · Google figures out how to watermark AI-designed proteins — _The article focuses on Google DeepMind’s research adapting SynthID to watermark AI-designed protein sequences for biosecurity._
- 0.449 · Google thinks SpaceX’s Starship has to launch 1,800 times before space data centers get off the ground — _Google is testing TPUs in orbit and developing Project Suncatcher, an AI-focused orbital data center initiative supported by new research._
- 0.455 · Google researchers find a way to keep self-improving AI agents from memorizing their tests — _Google Cloud AI Research developed RRSI, a method that improves AI-agent harnesses while preventing benchmark memorization and reducing runtime compute._
- 0.456 · Judge dismisses Chegg and Penske antitrust lawsuits targeting Google AI search — _The article centers on Google’s AI Overviews and Gemini models, and a judge’s dismissal of antitrust lawsuits over their content use and traffic impact._
- 0.470 · Google's early attempt to pay websites for AI answers is struggling — _Google’s AI contribution pilot pays publishers whose content supports Gemini-powered Search answers, but reported payouts are mostly minuscule._
- 0.477 · AI beats Stratego's greatest player, ending one of the last human strongholds in board games — _The article substantially discusses DeepMind’s DeepNash Stratego system and its TPU training costs, comparing it with a new competing AI._
- 0.489 · Trump plan to combat AI risks hinges on Big Tech pals policing themselves — _Google is a signatory to the White House’s voluntary AI safety agreement and had already committed to external audits of its AI systems._

**False positives** (candidates the judge rejected): 31 — keyword only 11, semantic only 6, both 4, ambiguous name 10

**Semantic-only false positives** (6)

- 0.533 · Open-source "BootLoops" harness supports AI models in performing precise scientific calculations — _The article focuses on Anthropic’s Claude and Harvard’s BootLoops, with no Google AI work or products as a main subject._
- 0.528 · OpenAI launches visual ads that appear alongside image generation results — _The article focuses entirely on OpenAI’s ChatGPT advertising features and does not discuss Google’s AI work._
- 0.523 · It’s not AI anymore, it’s ‘super intelligence’ (according to the White House) — _The article discusses White House policy, Meta, OpenAI, and broader AI economics; it does not mention or cover Google’s AI work._
- 0.520 · Call it AI, call it Super Intelligence, only 2% of consumers are buying it — _The article discusses broad AI industry developments involving Meta, OpenAI, Anthropic and others; Google’s AI work is not mentioned._
- 0.519 · Can Safeworld convince people that GenAI robots won’t hurt them? — _The article focuses on Safeworld’s generative-AI robot safety startup; Google and its AI work are not mentioned._
- 0.509 · Get all your questions answered at TechCrunch Disrupt 2026: The full breakout session agenda revealed — _The article promotes TechCrunch Disrupt sessions about AI, but does not discuss any Google, Google DeepMind, or Google Labs AI work._

## Judge precision: manual spot check

Reviewed by hand on 2026-10-07: every accepted article (title, judge reason) and every
rejected candidate's title, for both experiments. "Wrong" means it fails the judge's own
rubric (the company's AI work must be a main subject); "borderline" could go either way.

**OpenAI (34 accepted):** about 29 clearly right, 2 borderline, 3 wrong, so judge
precision is roughly 85-90%.
- Wrong: *Opus 5.5 loves to tell you 'this matters'* (writing tells across many models),
  *Brian Chesky interview* (Airbnb CEO; ChatGPT's app store is an aside), *Hot Girl
  Hotline* (relationship-advice service; ChatGPT is context).
- Borderline: *Lawyer cites ChatGPT-invented fake witnesses* (ChatGPT is the subject, but
  OpenAI did nothing), *Google drops Gems for Skills* (mostly a Google story).
- No wrong rejections spotted among the 49 rejected candidates: Gemini, Mistral, Anthropic
  and Meta stories were all correctly turned down despite naming OpenAI.

**Google (22 accepted candidates + 3 audit finds):** about 18 clearly right, 3 borderline, 4
wrong, so judge precision is roughly 80-85%.
- Wrong: *Opus 5.5 loves to tell you 'this matters'* again (accepted by both judges for
  the same passing mention), *AutoSynthData* and *Satlyt* (third parties using Gemma; the
  rubric asks for Google's own work), *Internet Infrastructure Services Empower Deepfake
  Abuse* (audit: about Google's SSL certificates and ads, not its AI).
- Borderline: *Trump plan to combat AI risks* (Google is one signatory among many), *Google
  froze its open source bug bounty program* (audit: Google reacting to AI-generated reports,
  not shipping AI), *MCP for agent-to-agent comms* (audit: Google acknowledging flaws in its
  agent systems, a small part of the article).
- No wrong rejections spotted among the 31 rejected candidates.

**Where the judge goes wrong** is the same in both: it accepts articles where a product is
the topic or the tool but the company did nothing (ChatGPT misuse, others building on
Gemma), and it accepted one multi-model article for both companies. Tightening the rubric
on "used by others" and "one of several models" is the obvious next prompt change.

