---
name: multi-hop-qa-tips
description: "For multi-hop QA -- chain entity searches step by step, don't search the full question"
task-type: multi_hop_qa
source: curated
---

Chain entities step by step: search entity A, extract link to B, search B, then answer.
Pitfall: Don't search the full question. Break into single-entity queries.
If lookup returns NO_MATCH, try a shorter keyword or alternate spelling.
