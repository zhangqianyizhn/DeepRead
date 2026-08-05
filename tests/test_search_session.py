import json
import unittest
from unittest.mock import patch

from DeepRead.agent.runner import run_agent
from DeepRead.agent.search_session import (
    RetrievalStrategyState,
    SearchSessionState,
    chunk_id_for_result,
    normalize_requested_top_k,
)
from DeepRead.tool.retrieval import DocIndex
from DeepRead.tool.schema import make_tools_schema


def _hit(paragraph_index: int, text: str) -> dict:
    return {
        "score": 1.0,
        "ref": {
            "doc_id": "7",
            "node_id": "12",
            "hit_paragraph_index": paragraph_index,
            "paragraph_indexes": [paragraph_index - 1, paragraph_index],
        },
        "text": text,
        "neighbors": [],
    }


class SearchSessionStateTest(unittest.TestCase):
    def test_repeated_search_channel_gets_visible_stagnation_warning(self) -> None:
        state = RetrievalStrategyState(stagnation_threshold=3)
        outputs = []
        for paragraph_index in range(3):
            outputs.append(
                state.annotate(
                    {"ok": True, "results": [_hit(paragraph_index, "new text")]},
                    tool_name="bm25_search",
                    scope="full",
                    doc_id=None,
                )
            )

        self.assertEqual(outputs[1]["retrieval_strategy"]["status"], "CONTINUE")
        warning = outputs[2]["retrieval_strategy"]
        self.assertEqual(warning["status"], "STRATEGY_STAGNATION")
        self.assertEqual(warning["same_channel_streak"], 3)
        self.assertIn("Switch retrieval method", warning["recommended_actions"][0])

    def test_agent_top_k_is_clamped(self) -> None:
        self.assertEqual(normalize_requested_top_k(None, 1, 10), 1)
        self.assertEqual(normalize_requested_top_k("6", 1, 10), 6)
        self.assertEqual(normalize_requested_top_k(99, 1, 10), 10)
        self.assertEqual(normalize_requested_top_k(0, 1, 10), 1)

    def test_repeat_hits_are_marked_and_lower_ranked_hits_fill_page(self) -> None:
        state = SearchSessionState()
        first = state.paginate(
            {"ok": True, "results": [_hit(1, "one"), _hit(2, "two")]},
            requested_top_k=2,
            round_id=1,
            candidate_top_k=2,
        )
        self.assertEqual([r["text"] for r in first["results"]], ["one", "two"])
        self.assertTrue(all(r["status"] == "NEW_RESULT" for r in first["results"]))

        candidate_top_k = state.candidate_top_k(2, 50)
        self.assertEqual(candidate_top_k, 4)
        second = state.paginate(
            {
                "ok": True,
                "results": [
                    _hit(1, "one"),
                    _hit(2, "two"),
                    _hit(3, "three"),
                    _hit(4, "four"),
                ],
            },
            requested_top_k=2,
            round_id=2,
            candidate_top_k=candidate_top_k,
        )
        self.assertEqual(
            [r.get("text") for r in second["results"] if r["status"] == "NEW_RESULT"],
            ["three", "four"],
        )
        seen_results = [
            r
            for r in second["results"]
            if r["status"] == "ALREADY_SEEN_FULL_TEXT_AVAILABLE_IN_HISTORY"
        ]
        self.assertEqual(len(seen_results), 2)
        self.assertEqual(
            seen_results[0]["status"],
            "ALREADY_SEEN_FULL_TEXT_AVAILABLE_IN_HISTORY",
        )
        self.assertEqual(second["pagination"]["returned_new_results"], 2)

    def test_tool_schema_exposes_optional_agent_top_k(self) -> None:
        fake_index = type("FakeIndex", (), {"neighbor_window": None})()
        tools = make_tools_schema(
            fake_index, enable_semantic=True, suggested_top_k=1, max_top_k=8
        )
        search_tools = [
            tool["function"]
            for tool in tools
            if tool["function"]["name"].endswith("search")
            or tool["function"]["name"] == "semantic_retrieval"
        ]
        self.assertEqual(len(search_tools), 5)
        for tool in search_tools:
            top_k = tool["parameters"]["properties"]["top_k"]
            self.assertEqual(top_k["default"], 1)
            self.assertEqual(top_k["maximum"], 8)
            self.assertNotIn("top_k", tool["parameters"]["required"])

    def test_chunk_identity_is_stable_across_search_methods(self) -> None:
        index = DocIndex(
            [
                {
                    "doc_id": "1",
                    "id": "2",
                    "title": "Acquisitions",
                    "paragraphs": ["The acquisition consideration was $10 million."],
                    "children": [],
                }
            ],
            neighbor_window=(0, 0),
        )
        bm25_hit = index.bm25_search("acquisition", top_k=1)["results"][0]
        regex_hit = index.regex_search("acquisition", top_k=1)["results"][0]
        self.assertEqual(chunk_id_for_result(bm25_hit), "1/2/0")
        self.assertEqual(
            chunk_id_for_result(bm25_hit), chunk_id_for_result(regex_hit)
        )

    def test_runner_uses_agent_top_k_and_visible_pagination(self) -> None:
        class FakeIndex:
            neighbor_window = None
            nodes_by_doc = {"7": {}}

            def __init__(self) -> None:
                self.requested_sizes = []

            def bm25_search(self, *, top_k: int, **kwargs) -> dict:
                self.requested_sizes.append(top_k)
                return {
                    "ok": True,
                    "results": [
                        _hit(index, f"text-{index}") for index in range(1, top_k + 1)
                    ],
                }

        class FakeLogger:
            def log(self, *args, **kwargs) -> None:
                pass

        tool_call = {
            "type": "function",
            "function": {
                "name": "bm25_search",
                "arguments": '{"query":"revenue","scope":"full","top_k":2}',
            },
        }
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [{**tool_call, "id": "call-1"}],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [{**tool_call, "id": "call-2"}],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [{**tool_call, "id": "call-3"}],
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "done"}}]},
        ]
        sent_payloads = []

        def fake_chat(**kwargs):
            sent_payloads.append(kwargs["payload"])
            return responses.pop(0)

        index = FakeIndex()
        with patch(
            "DeepRead.agent.runner.http_chat_completions", side_effect=fake_chat
        ):
            answer = run_agent(
                model="test",
                base_url=None,
                doc_index=index,
                user_question="question",
                logger=FakeLogger(),
                max_rounds=4,
                disable_regex=True,
                agent_topk_max=10,
                pagination_candidate_limit=50,
            )

        self.assertEqual(answer, "done")
        self.assertEqual(index.requested_sizes, [2, 4, 6])
        tool_messages = [
            message
            for message in sent_payloads[-1]["messages"]
            if message["role"] == "tool"
        ]
        second_page = json.loads(tool_messages[-2]["content"])
        seen_results = [
            result
            for result in second_page["results"]
            if result["status"] == "ALREADY_SEEN_FULL_TEXT_AVAILABLE_IN_HISTORY"
        ]
        self.assertEqual(len(seen_results), 2)
        self.assertEqual(
            [
                result["text"]
                for result in second_page["results"]
                if result["status"] == "NEW_RESULT"
            ],
            ["text-3", "text-4"],
        )
        third_page = json.loads(tool_messages[-1]["content"])
        self.assertEqual(
            third_page["retrieval_strategy"]["status"],
            "STRATEGY_STAGNATION",
        )


if __name__ == "__main__":
    unittest.main()
