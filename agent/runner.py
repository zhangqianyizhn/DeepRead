from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from .llm import (
    _preview_messages,
    _preview_tool_calls,
    detect_provider,
    http_chat_completions,
    sanitize_for_vllm,
    should_sanitize_for_vllm,
)
from ..prompt.system import build_system_prompt
from ..tool.fallback import fallback_tool_calls_from_text, strip_function_calls_block_any, strip_inline_tool_calls
from ..tool.schema import make_tools_schema
from .search_session import (
    RetrievalStrategyState,
    SearchSessionState,
    normalize_requested_top_k,
)


def run_agent(
    model: str,
    base_url: Optional[str],
    doc_index: DocIndex,
    user_question: str,
    logger: JsonlLogger,
    max_rounds: int = 50,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None,
    enable_multimodal: bool = False,
    enable_vector: bool = False,
    enable_hybrid: bool = False,
    enable_semantic: bool = False,
    disable_bm25: bool = False,
    disable_regex: bool = False,
    disable_read: bool = False,
    embed_api_key: Optional[str] = None,
    embed_base_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
    neighbor_window: Optional[Tuple[int, int]] = None,
    bm25_topk: int = 1,
    regex_topk: int = 1,
    vector_topk: int = 1,
    hybrid_topk: int = 1,
    hybrid_topk_bm25: int = 30,
    hybrid_topk_vec: int = 30,
    hybrid_bm25_weight: float = 0.5,
    hybrid_vector_weight: float = 0.5,
    semantic_stage1_method: str = "vector",
    semantic_topk1: int = 30,
    semantic_topk2: int = 1,
    semantic_stage1_hybrid_topk_bm25: int = 30,
    semantic_stage1_hybrid_topk_vec: int = 30,
    rerank_api_key: Optional[str] = None,
    rerank_base_url: str = "https://api.siliconflow.cn/v1",
    rerank_model: str = "Qwen/Qwen3-Reranker-8B",
    tool_fallback: bool = True,
    enable_reasoning: bool = True,
    collected_texts: Optional[List[str]] = None,
    enable_session_pagination: bool = True,
    agent_topk_max: int = 10,
    pagination_candidate_limit: int = 50,
    enable_retrieval_stagnation_hint: bool = True,
    retrieval_stagnation_threshold: int = 3,
    additional_instructions: Optional[List[str] | str] = None,
    reasoning_effort: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
    request_timeout: int = 120,
) -> str:
    tools = make_tools_schema(
        doc_index,
        enable_semantic=enable_semantic,
        suggested_top_k=bm25_topk,
        max_top_k=agent_topk_max,
    )
    query_id = hashlib.sha1(user_question.encode('utf-8')).hexdigest()[:16]

    if disable_bm25:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "bm25_search"]
    if disable_regex:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "regex_search"]
    if not enable_vector:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "vector_search"]
    if not enable_hybrid:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "hybrid_search"]
    if not enable_semantic:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "semantic_retrieval"]
    if disable_read:
        tools = [t for t in tools if (t.get("function") or {}).get("name") != "read_section"]

    tool_names = [(t.get("function") or {}).get("name") for t in tools]
    system_prompt = build_system_prompt(
        doc_index,
        tool_names,
        enable_reasoning=enable_reasoning,
        additional_instructions=additional_instructions,
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}]
    prev_msg_count = 1
    search_session = SearchSessionState()
    retrieval_strategy = RetrievalStrategyState(
        stagnation_threshold=retrieval_stagnation_threshold
    )
    search_tool_names = {
        "bm25_search",
        "regex_search",
        "vector_search",
        "hybrid_search",
        "semantic_retrieval",
    }

    do_sanitize = should_sanitize_for_vllm(base_url)
    provider = detect_provider(model, base_url)
    # Kimi（Moonshot）保留式思考与 DeepSeek 思考模式（携带 tools 时）都要求
    # 历史 assistant 消息以 reasoning_content 字段原样回传，否则 DeepSeek 会报 400；
    # 火山引擎 Ark 使用 reasoning 字段
    reasoning_history_key = "reasoning_content" if provider in ("kimi", "deepseek") else "reasoning"

    effective_neighbor_window: Optional[Tuple[int, int]] = neighbor_window if neighbor_window is not None else doc_index.neighbor_window

    for round_id in range(1, max_rounds + 1):
        req_payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
        }
        if provider == "kimi":
            # kimi-k3 始终思考：不支持 include_reasoning，temperature 不可修改（不传）。
            # 思考强度通过顶层 reasoning_effort（low/high/max，默认 max）配置；
            # 上下文窗口为模型固定（kimi-k3 为 1M tokens），仅可用 max_completion_tokens 限制输出长度。
            if reasoning_effort:
                req_payload["reasoning_effort"] = reasoning_effort
            if max_completion_tokens:
                req_payload["max_completion_tokens"] = max_completion_tokens
        elif provider == "deepseek":
            # DeepSeek-V4：思考模式默认开启，用 thinking.type 显式声明；
            # 思考强度 reasoning_effort（low/high/max，默认 high）；
            # 思考模式下 temperature/top_p 等采样参数无效，故不发送（关闭思考时才发送）；
            # 输出长度上限参数名为 max_tokens（非 max_completion_tokens）。
            req_payload["thinking"] = {"type": "enabled" if enable_reasoning else "disabled"}
            if reasoning_effort:
                req_payload["reasoning_effort"] = reasoning_effort
            if max_completion_tokens:
                req_payload["max_tokens"] = max_completion_tokens
            if not enable_reasoning:
                req_payload["temperature"] = temperature
        else:
            req_payload["temperature"] = temperature
            req_payload["include_reasoning"] = bool(enable_reasoning)

        logger.log("llm_request", query_id=query_id, round=round_id, base_url=base_url, context_delta_preview=_preview_messages(messages[prev_msg_count:]))
        prev_msg_count = len(messages)

        payload_to_send = sanitize_for_vllm(req_payload, allow_tools=True) if do_sanitize else req_payload

        try:
            resp = http_chat_completions(query_id=query_id, api_key=api_key, base_url=base_url, payload=payload_to_send, default_headers=default_headers, timeout=request_timeout, logger=logger)
        except Exception as exc:
            logger.log("llm_http_error", query_id=query_id, error=str(exc), round=round_id)
            resp = {}

        msg = (resp.get("choices") or [{}])[0].get("message", {})  # type: ignore

        reasoning_content = None
        if enable_reasoning:
            for rf in ["reasoning", "reasoning_content", "thinking", "internal_monologue"]:
                if msg.get(rf) is not None:
                    reasoning_content = msg.get(rf)
                    break

        tool_calls = msg.get("tool_calls")
        recovered_from_text = False
        recovered_meta: Dict[str, Any] = {}

        content_str = msg.get("content") or ""
        reasoning_str = reasoning_content or ""

        if not tool_calls and tool_fallback:
            rec = fallback_tool_calls_from_text(content_str) or fallback_tool_calls_from_text(reasoning_str)
            if rec:
                tool_calls, recovered_meta = rec
                recovered_from_text = True
                logger.log(
                    "tool_calls_recovered_from_text",
                    query_id=query_id,
                    round=round_id,
                    recovered=_preview_tool_calls(tool_calls),
                    recovered_kind=recovered_meta.get("kind"),
                )

        assistant_entry: Dict[str, Any] = {"role": "assistant"}

        content_for_history = msg.get("content")
        if isinstance(content_for_history, str) and recovered_from_text:
            content_for_history = strip_function_calls_block_any(content_for_history)
            if recovered_meta.get("kind") == "inline_json" and recovered_meta.get("spans"):
                content_for_history = strip_inline_tool_calls(content_for_history, recovered_meta["spans"])

        if content_for_history is None:
            content_for_history = ""
        assistant_entry["content"] = content_for_history

        if enable_reasoning and reasoning_content:
            assistant_entry[reasoning_history_key] = reasoning_content

        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls

        messages.append(assistant_entry)

        logger.log(
            "llm_response",
            query_id=query_id,
            round=round_id,
            content=msg.get("content"),
            reasoning_content=reasoning_content if enable_reasoning else None,
            tool_calls=_preview_tool_calls(tool_calls),
            context_delta_preview=_preview_messages(messages[prev_msg_count:]) if tool_calls else None,
        )

        if tool_calls:
            prev_msg_count = len(messages)

        if not tool_calls:
            final_answer = (msg.get("content") or "").strip()
            if final_answer:
                logger.log("final_answer", query_id=query_id, answer=final_answer, context_delta_preview=_preview_messages(messages[prev_msg_count:]))
                return final_answer

            if enable_reasoning and (reasoning_content is not None) and str(reasoning_content).strip():
                logger.log("llm_thinking_only", query_id=query_id, round=round_id, reasoning_preview=str(reasoning_content)[:2000])
            else:
                logger.log("llm_empty_message", query_id=query_id, round=round_id)
            continue

        for tc in tool_calls or []:
            tool_name = (tc.get("function") or {}).get("name")
            try:
                args_raw = (tc.get("function") or {}).get("arguments")
                args = args_raw if isinstance(args_raw, dict) else json.loads(args_raw or "{}")
            except Exception as exc:
                logger.log("tool_args_parse_error", query_id=query_id, tool=tool_name, raw=str(args_raw), error=str(exc))
                args = {}

            logger.log("tool_call", query_id=query_id, tool=tool_name, args=args, tool_call_id=tc.get("id"))

            try:
                requested_top_k: Optional[int] = None
                candidate_top_k: Optional[int] = None
                if tool_name == "get_doc_structure":
                    raw_ids = args.get("doc_id")
                    doc_ids = [str(d) for d in raw_ids] if isinstance(raw_ids, list) else None
                    out = doc_index.get_doc_structure(doc_ids=doc_ids)
                elif tool_name == "read_section":
                    out = doc_index.read_section(
                        doc_id=args.get("doc_id"),
                        node_id=args.get("node_id"),
                        start_paragraph=int(args.get("start_paragraph", 0)),
                        end_paragraph=int(args.get("end_paragraph", -1)),
                        include_images=enable_multimodal,
                    )
                elif tool_name == "bm25_search":
                    requested_top_k = normalize_requested_top_k(
                        args.get("top_k"), bm25_topk, agent_topk_max
                    )
                    candidate_top_k = (
                        search_session.candidate_top_k(
                            requested_top_k, pagination_candidate_limit
                        )
                        if enable_session_pagination
                        else requested_top_k
                    )
                    out = doc_index.bm25_search(
                        query=args.get("query", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        top_k=candidate_top_k,
                        include_images=enable_multimodal,
                        neighbor_window=effective_neighbor_window,
                    )
                elif tool_name == "regex_search":
                    requested_top_k = normalize_requested_top_k(
                        args.get("top_k"), regex_topk, agent_topk_max
                    )
                    candidate_top_k = (
                        search_session.candidate_top_k(
                            requested_top_k, pagination_candidate_limit
                        )
                        if enable_session_pagination
                        else requested_top_k
                    )
                    out = doc_index.regex_search(
                        pattern=args.get("pattern", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        top_k=candidate_top_k,
                        include_images=enable_multimodal,
                        neighbor_window=effective_neighbor_window,
                    )
                elif tool_name == "vector_search":
                    requested_top_k = normalize_requested_top_k(
                        args.get("top_k"), vector_topk, agent_topk_max
                    )
                    candidate_top_k = (
                        search_session.candidate_top_k(
                            requested_top_k, pagination_candidate_limit
                        )
                        if enable_session_pagination
                        else requested_top_k
                    )
                    out = doc_index.vector_search(
                        query=args.get("query", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        top_k=candidate_top_k,
                        include_images=enable_multimodal,
                        embed_api_key=embed_api_key,
                        embed_base_url=embed_base_url,
                        embed_model=embedding_model,
                        neighbor_window=effective_neighbor_window,
                    )
                elif tool_name == "hybrid_search":
                    requested_top_k = normalize_requested_top_k(
                        args.get("top_k"), hybrid_topk, agent_topk_max
                    )
                    candidate_top_k = (
                        search_session.candidate_top_k(
                            requested_top_k, pagination_candidate_limit
                        )
                        if enable_session_pagination
                        else requested_top_k
                    )
                    out = doc_index.hybrid_search(
                        query=args.get("query", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        top_k=candidate_top_k,
                        bm25_weight=float(hybrid_bm25_weight),
                        vector_weight=float(hybrid_vector_weight),
                        top_k_bm25=max(int(hybrid_topk_bm25), candidate_top_k),
                        top_k_vec=max(int(hybrid_topk_vec), candidate_top_k),
                        include_images=enable_multimodal,
                        embed_api_key=embed_api_key,
                        embed_base_url=embed_base_url,
                        embed_model=embedding_model,
                        neighbor_window=effective_neighbor_window,
                    )
                elif tool_name == "semantic_retrieval":
                    requested_top_k = normalize_requested_top_k(
                        args.get("top_k"), semantic_topk2, agent_topk_max
                    )
                    candidate_top_k = (
                        search_session.candidate_top_k(
                            requested_top_k, pagination_candidate_limit
                        )
                        if enable_session_pagination
                        else requested_top_k
                    )
                    out = doc_index.semantic_retrieval(
                        query=args.get("query", ""),
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                        stage1_method=str(semantic_stage1_method),
                        top_k1=max(int(semantic_topk1), candidate_top_k),
                        top_k2=candidate_top_k,
                        stage1_hybrid_topk_bm25=int(semantic_stage1_hybrid_topk_bm25),
                        stage1_hybrid_topk_vec=int(semantic_stage1_hybrid_topk_vec),
                        include_images=enable_multimodal,
                        embed_api_key=embed_api_key,
                        embed_base_url=embed_base_url,
                        embed_model=embedding_model,
                        rerank_api_key=rerank_api_key,
                        rerank_base_url=rerank_base_url,
                        rerank_model=rerank_model,
                        neighbor_window=effective_neighbor_window,
                        hybrid_bm25_weight=float(hybrid_bm25_weight),
                        hybrid_vector_weight=float(hybrid_vector_weight),
                    )
                else:
                    out = {"ok": False, "error": f"Tool '{tool_name}' not implemented"}

                if (
                    enable_session_pagination
                    and tool_name in search_tool_names
                    and requested_top_k is not None
                    and candidate_top_k is not None
                ):
                    out = search_session.paginate(
                        out,
                        requested_top_k=requested_top_k,
                        round_id=round_id,
                        candidate_top_k=candidate_top_k,
                    )

                if (
                    enable_retrieval_stagnation_hint
                    and tool_name in search_tool_names
                ):
                    out = retrieval_strategy.annotate(
                        out,
                        tool_name=tool_name,
                        scope=args.get("scope", "full"),
                        doc_id=args.get("doc_id"),
                    )

                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(out, ensure_ascii=False)})

                if collected_texts is not None and isinstance(out, dict) and out.get("ok", True):
                    if tool_name in ("bm25_search", "regex_search", "vector_search", "hybrid_search", "semantic_retrieval"):
                        for r in out.get("results", []):
                            if r.get("text"):
                                collected_texts.append(r["text"])
                            for nb in r.get("neighbors", []):
                                if isinstance(nb, dict) and nb.get("type") == "text" and nb.get("text"):
                                    collected_texts.append(nb["text"])
                    elif tool_name == "read_section":
                        for p in out.get("paragraphs", []):
                            if isinstance(p, dict) and p.get("type") == "text" and p.get("text"):
                                collected_texts.append(p["text"])
                
                if (
                    enable_multimodal
                    and tool_name in ("bm25_search", "regex_search", "vector_search", "hybrid_search", "semantic_retrieval")
                    and isinstance(out, dict)
                ):
                    seen_keys = set()
                    mm_items: List[Dict[str, Any]] = []
                    for r in out.get("results", []):
                        neighbors = r.get("neighbors") or []
                        ref = r.get("ref") or {}
                        did = str(ref.get("doc_id"))
                        nid = str(ref.get("node_id"))
                        for item in neighbors:
                            if isinstance(item, dict):
                                item_type = item.get("type") or "text"
                                par_idx = int(item.get("paragraph_index", -1))
                                key = (did, nid, par_idx, item_type)
                                if key in seen_keys:
                                    continue
                                seen_keys.add(key)
                                tagged = dict(item)
                                tagged["paragraph_index"] = par_idx
                                mm_items.append(tagged)
                    if mm_items:
                        messages.append({"role": "user", "content": mm_items})

                logger.log(
                    "tool_result",
                    query_id=query_id,
                    tool=tool_name,
                    ok=bool(out.get("ok", True)) if isinstance(out, dict) else True,
                    result=out,
                    context_delta_preview=_preview_messages(messages[prev_msg_count:]),
                )
                prev_msg_count = len(messages)

            except Exception as exc:
                err = {"ok": False, "error": str(exc)}
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": json.dumps(err, ensure_ascii=False)})
                logger.log(
                    "tool_result",
                    query_id=query_id,
                    tool=tool_name,
                    ok=False,
                    error=str(exc),
                    result=err,
                    context_delta_preview=_preview_messages(messages[prev_msg_count:]),
                )
                prev_msg_count = len(messages)

    logger.log("max_rounds_reached", query_id=query_id, max_rounds=max_rounds)
    return "(Reached maximum rounds, no final answer generated)"
