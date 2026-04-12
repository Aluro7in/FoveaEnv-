# grader.py
# FoveaEnv — Scoring Logic (FINAL VERSION)

import math

MIN_SCORE = 0.0001
MAX_SCORE = 0.9999


def _strict_score(x: float) -> float:
    """
    Clamp score strictly into (0, 1) and round to 4 decimals.
    Guarantees output is always in (0.0001, 0.9999) — never 0.0 or 1.0.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.5

    if not math.isfinite(x):
        return 0.5

    return round(max(MIN_SCORE, min(MAX_SCORE, x)), 4)


def _safe_nav(episode_reward: float) -> float:
    """
    Map episode_reward into (0, 1).
    episode_reward range in FoveaEnv is roughly (-inf, ~2.0).
    We normalise with a sigmoid so it is always strictly in (0, 1)
    regardless of any reward value the environment produces.
    """
    import math as _math
    try:
        r = float(episode_reward)
    except (TypeError, ValueError):
        r = 0.0
    if not _math.isfinite(r):
        r = 0.0
    # Sigmoid centred at 0, scaled so reward=0 -> 0.5
    return 1.0 / (1.0 + _math.exp(-r))


def _safe_priv(privacy_violations: int, total_steps: int) -> float:
    """
    Privacy efficiency in (0, 1).
    Fewer violations relative to steps is better.
    Uses exponential decay so the result is never exactly 0 or 1.
    """
    try:
        v = int(privacy_violations) if privacy_violations is not None else 0
    except (TypeError, ValueError):
        v = 0
    try:
        s = int(total_steps) if total_steps is not None else 1
    except (TypeError, ValueError):
        s = 1
    s = max(s, 1)
    v = max(v, 0)
    # violation_rate in [0, inf); exp(-rate) maps it strictly into (0, 1]
    # We then clamp with _strict_score to keep it out of exact 0 or 1.
    rate = v / s
    return math.exp(-rate)


def grade_episode(
    episode_reward: float,
    reached_goal: bool,
    privacy_violations: int,
    total_steps: int,
) -> dict:
    """
    Compute final scores for an episode.

    All three returned scores are guaranteed to be strictly between 0 and 1
    (i.e. in the open interval (0, 1), never 0.0 and never 1.0).

    Notes:
    - 'reached_goal' is accepted for compatibility, but does not directly alter scoring.
    - Final score stays strictly between 0 and 1.
    - 'reached_goal' is accepted for compatibility but does not alter scoring.
    - Navigation uses a sigmoid so extreme rewards never hit 0 or 1.
    - Privacy uses exponential decay so 0 violations never returns exactly 1.
    """
    try:
        episode_reward = float(episode_reward) if episode_reward is not None else 0.0
    except (TypeError, ValueError):
        episode_reward = 0.0

    try:
        privacy_violations = (
            int(privacy_violations) if privacy_violations is not None else 0
        )
    except (TypeError, ValueError):
        privacy_violations = 0

    try:
        total_steps = int(total_steps) if total_steps is not None else 1
    except (TypeError, ValueError):
        total_steps = 1

    total_steps = max(total_steps, 1)

    # Navigation quality: maps episode reward into a normalized score.
    raw_nav = (episode_reward + 0.5) / 2.5

    # Privacy efficiency: fewer violations over fewer steps is better.
    raw_priv = 1.0 - (privacy_violations / total_steps)

    # Final score is a fixed weighted blend.
    raw_nav = _safe_nav(episode_reward)
    raw_priv = _safe_priv(privacy_violations, total_steps)
    raw_final = 0.6 * raw_nav + 0.4 * raw_priv

    return {
        "navigation_score": _strict_score(raw_nav),
        "privacy_efficiency_score": _strict_score(raw_priv),
        "final_score": _strict_score(raw_final),
        # Final score is a fixed weighted blend — always strictly in (0, 1).
    }
