# grader.py
# FoveaEnv — Scoring Logic (FINAL VERSION)

import math

MIN_SCORE = 0.0001
MAX_SCORE = 0.9999


def _strict_score(x: float) -> float:
    """
    Clamp score strictly into (0, 1) and round to 4 decimals.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.5

    if not math.isfinite(x):
        return 0.5

    return round(max(MIN_SCORE, min(MAX_SCORE, x)), 4)


def grade_episode(
    episode_reward: float,
    reached_goal: bool,
    privacy_violations: int,
    total_steps: int,
) -> dict:
    """
    Compute final scores for an episode.

    Notes:
    - 'reached_goal' is accepted for compatibility, but does not directly alter scoring.
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

    # Navigation quality: maps episode reward into a normalized score.
    raw_nav = (episode_reward + 0.5) / 2.5

    # Privacy efficiency: fewer violations over fewer steps is better.
    raw_priv = 1.0 - (privacy_violations / total_steps)

    # Final score is a fixed weighted blend.
    raw_final = 0.6 * raw_nav + 0.4 * raw_priv

    return {
        "navigation_score": _strict_score(raw_nav),
        "privacy_efficiency_score": _strict_score(raw_priv),
        "final_score": _strict_score(raw_final),
    }
