from __future__ import annotations

from typing import List, Optional, Union


# ------------------------------
# Agent loop helpers
# ------------------------------
def build_system_prompt(
    doc_index: DocIndex,
    tool_names: List[str],
    enable_reasoning: bool = True,
    additional_instructions: Optional[Union[List[str], str]] = None,
) -> str:
    total_docs = len(doc_index.nodes_by_doc)
    search_tools = [t for t in tool_names if ("search" in (t or "")) or ("retrieval" in (t or ""))]
    search_cmd = f"Use {', '.join(search_tools)}" if search_tools else "Search"

    constraints = [
        f"{search_cmd} to locate relevant content across documents or nodes of specific document based on Directory Structure.",
        "After finding results, if you have not yet obtained the Directory Structure for the relevant document(s), call get_doc_structure with the relevant doc_id(s) to inspect it before calling read_section",
        "Answer strictly based on the provided corpus; do not fabricate.",
        "Parsing errors may cause body text to be mistakenly treated as hierarchical elements (or headings), rendering the heading text inaccessible to search and reading tools. Please make reasonable inferences based on the Directory Structure and the content returned by the tool.",
        "Respond in the User's language; align queries with the Directory Structure.",
        "Usually, you need to think step by step and then call tools to locate or get structure or read, iterating in this way until you can answer the question.",
        "Search results preserve backend rank. Entries marked NEW_RESULT contain full text; repeated chunks appear in the same results list as compact ALREADY_SEEN_FULL_TEXT_AVAILABLE_IN_HISTORY markers, followed by lower-ranked new chunks when available.",
        "If a search mostly repeats earlier chunks, change the query, retrieval method, scope, or document; inspect a promising document structure; increase top_k when broader recall is needed; or answer once the evidence is sufficient.",
        "When calling tools, DO NOT write tool invocations in plain text. Use the structured tool call interface (tool_calls) only.",
    ]

    constraints_block = "\n".join(f"- {c}" for c in constraints)
    if isinstance(additional_instructions, str):
        extra_items = [additional_instructions.strip()] if additional_instructions.strip() else []
    else:
        extra_items = [
            str(item).strip()
            for item in (additional_instructions or [])
            if str(item).strip()
        ]
    extra_block = ""
    if extra_items:
        rendered = "\n".join(f"- {item}" for item in extra_items)
        extra_block = f"\n\n## Additional task instructions\n{rendered}"

    return (
        f"You are a documents assistant. The corpus contains {total_docs} document(s), "
        f"with doc_id values ranging from 1 to {total_docs}.\n\n"
        "The Directory Structure of each document lists all its nodes in the format:\n"
        "`- (doc_id) [node_id] Title | paragraphs=Num | tokens=Num | "
        "children=[ID list]`.\n"
        "You can use get_doc_structure to retrieve the Directory Structure of "
        "specific documents.\n"
        "Use this structure and your available tools to answer the user's question.\n\n"
        f"## Guidelines\n{constraints_block}{extra_block}"
    )
