from src.utils.llm_router import extract_json_from_response, score_candidate_margin, should_escalate


class TestLLMRouterHelpers:
    def test_extract_json_from_response_parses_fenced_json(self):
        text = '```json\n{"a": 1, "b": 2}\n```'
        assert extract_json_from_response(text) == {"a": 1, "b": 2}

    def test_score_candidate_margin_uses_top_two_scores(self):
        candidates = [{"score": 0.91}, {"score": 0.86}, {"score": 0.4}]
        assert abs(score_candidate_margin(candidates) - 0.05) < 1e-9

    def test_should_escalate_when_confidence_low(self):
        assert should_escalate(
            cheap_confidence=0.7,
            candidate_margin=0.3,
            is_new_title=False,
            noisy_title=False,
            context_conflict=False,
            has_conflicting_candidates=False,
        )

    def test_should_not_escalate_for_clear_easy_case(self):
        assert not should_escalate(
            cheap_confidence=0.95,
            candidate_margin=0.2,
            is_new_title=False,
            noisy_title=False,
            context_conflict=False,
            has_conflicting_candidates=False,
        )
