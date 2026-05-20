


CURATED_STRATEGIES = {
    "multi_hop_qa": (
        "Chain entities step by step: search entity A → extract link to B → search B → answer.\n"
        "Pitfall: Don't search the full question. Break into single-entity queries.\n"
        "If lookup returns NO_MATCH, try a shorter keyword or alternate spelling."
    ),

    "factual_qa": (
        "One search usually suffices. Extract the answer from the first relevant passage.\n"
        "Use lookup to pinpoint the exact sentence if search returns a long passage.\n"
        "Pitfall: Don't over-verify simple facts — answer quickly after one good source."
    ),

    "math_reasoning": (
        "Use python_execute to compute — never do arithmetic mentally.\n"
        "Break complex problems into steps: define variables → write equations → solve numerically.\n"
        "Pitfall: Don't repeat the same code if it errors. Read the error, fix the logic, then retry."
    ),

    "science_qa": (
        "For MCQ: analyze the clinical/scientific scenario, then pick the best option.\n"
        "Use ask_llm to reason through the options if unsure.\n"
        "Pitfall: Don't search for every keyword — most medical/science MCQs are solvable from the question text alone."
    ),

    "code_generation": (
        "Start with list_files → search_code to locate the relevant file and function.\n"
        "Read the failing area with view_file, then edit_file to fix.\n"
        "Pitfall: After a successful edit, STOP. Don't keep searching. Review the code context shown and answer."
    ),

    "interactive_agent": (
        "ALFWorld: Read admissible actions carefully. Pick from the list. Navigate → find → pick up → place.\n"
        "WebShop: search_product → click product → select options → click 'Buy Now'.\n"
        "Pitfall: Don't use ask_llm or python_execute — these environments only respond to act/click/search_product."
    ),
}
