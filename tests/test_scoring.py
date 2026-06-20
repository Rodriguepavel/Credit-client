import numpy as np

from credit_default.scoring import build_score_scale, probability_to_score, score_to_probability


def test_credit_score_decreases_when_default_probability_increases():
    scale = build_score_scale(base_default_rate=0.22)
    probabilities = np.array([0.05, 0.20, 0.50])
    scores = probability_to_score(probabilities, scale)

    assert scores[0] > scores[1] > scores[2]


def test_score_probability_conversion_round_trip():
    scale = build_score_scale(base_default_rate=0.22)
    probabilities = np.array([0.05, 0.20, 0.50])
    scores = probability_to_score(probabilities, scale)
    recovered = score_to_probability(scores, scale)

    np.testing.assert_allclose(recovered, probabilities)
