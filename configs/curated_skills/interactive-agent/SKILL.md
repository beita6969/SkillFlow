---
name: interactive-agent-tips
description: "For ALFWorld/WebShop -- use only environment-specific tools (act/click/search_product)"
task-type: interactive_agent
source: curated
---

ALFWorld: Read admissible actions carefully. Pick from the list. Navigate, find object, pick up, place.
WebShop: search_product, click product, select options, click 'Buy Now'.
Pitfall: Don't use ask_llm or python_execute -- these environments only respond to act/click/search_product.
