"""Research gates must not silently optimize against the holdout set."""
from copy import deepcopy

from scripts.code_search_research import acceptance_gate, select_trial


def trial(label, development, holdout):
    def metrics(mrr):
        return {"cases":5, "mrr":mrr, "recall@5":.8, "recall@10":1.,
                "indexed_gold_fraction":1., "vector_gold_fraction":1.}
    return {"label":label, "parameters":{"candidates":100, "vector_weight":.6},
            "summary":{"hybrid":{"development":metrics(development), "holdout":metrics(holdout)}},
            "cases":[{"mode":"hybrid", "status":"ok"}]}


def test_holdout_quality_cannot_influence_parameter_selection():
    development_winner = trial("development-winner", .8, .2)
    holdout_winner = trial("holdout-winner", .7, .95)
    assert select_trial([holdout_winner, development_winner])["label"] == "development-winner"
    assert not acceptance_gate(holdout_winner, development_winner)["accepted"]


def test_missing_vector_coverage_blocks_acceptance_even_with_perfect_observed_scores():
    baseline = trial("baseline", .6, .6)
    candidate = trial("candidate", 1., 1.)
    candidate["summary"]["hybrid"]["holdout"]["vector_gold_fraction"] = .2
    gate = acceptance_gate(baseline, candidate)
    assert not gate["accepted"]
    assert "holdout:incomplete_vector_gold_fraction" in gate["reasons"]


def test_mrr_gain_does_not_hide_recall_regression():
    baseline = trial("baseline", .6, .6)
    candidate = trial("candidate", .8, .8)
    candidate["summary"]["hybrid"]["holdout"]["recall@5"] = .4
    assert "holdout:regression_recall@5" in acceptance_gate(baseline, candidate)["reasons"]


def test_omitted_or_invalid_cases_cannot_pass_the_gate():
    baseline = trial("baseline", .6, .6)
    candidate = deepcopy(baseline)
    candidate["cases"].append({"mode":"hybrid", "status":"invalid_gold"})
    assert "unscored_or_invalid_cases" in acceptance_gate(baseline, candidate)["reasons"]


def test_equally_poor_runs_do_not_pass_absolute_quality_thresholds():
    baseline = trial("poor-baseline", .1, .1)
    candidate = deepcopy(baseline)
    candidate["summary"]["hybrid"]["holdout"]["recall@5"] = .1
    gate = acceptance_gate(baseline, candidate)
    assert "holdout:below_minimum_mrr" in gate["reasons"]
    assert "holdout:below_minimum_recall5" in gate["reasons"]


def test_false_no_answer_results_block_a_strong_ranking_score():
    baseline = trial("baseline", .8, .8)
    candidate = deepcopy(baseline)
    candidate["summary"]["hybrid"]["holdout"]["no_answer_accuracy"] = 0.
    assert "holdout:no_answer_failures" in acceptance_gate(baseline, candidate)["reasons"]
