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
    try:
        r = float(episode_reward)
    except (TypeError, ValueError):
        r = 0.0
    if not math.isfinite(r):
        r = 0.0
    # Sigmoid centred at 0, scaled so reward=0 -> 0.5
    raw = 1.0 / (1.0 + math.exp(-r))
    return _strict_score(raw)


def _safe_priv(privacy_violations: int, total_steps: int) -> float:
    """
    Privacy efficiency strictly in (0, 1).
    Fewer violations relative to steps is better.
    Uses exponential decay — result is then clamped with _strict_score
    so it is never exactly 0.0 or 1.0 (e.g. exp(-0) = 1.0 is clamped to 0.9999).
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
    # violation_rate in [0, inf); exp(-rate) maps it into (0, 1]
    # _strict_score clamps away the exact boundary values (0.0 and 1.0).
    rate = v / s
    raw = math.exp(-rate)
    return _strict_score(raw)


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
    - 'reached_goal' is accepted for compatibility but does not alter scoring.
    - Navigation uses a sigmoid so extreme rewards never hit 0 or 1.
    - Privacy uses exponential decay; _strict_score prevents exact 0 or 1.
    - Final score stays strictly between 0 and 1.
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

    # Navigation quality: sigmoid maps episode reward into (0, 1).
    nav_score = _safe_nav(episode_reward)

    # Privacy efficiency: exponential decay, clamped strictly into (0, 1).
    priv_score = _safe_priv(privacy_violations, total_steps)

    # Final score: fixed weighted blend, clamped strictly into (0, 1).
    raw_final = 0.6 * nav_score + 0.4 * priv_score
    final_score = _strict_score(raw_final)

    return {
        "navigation_score": nav_score,
        "privacy_efficiency_score": priv_score,
        "final_score": final_score,
    }
