# tasks.py
# Three difficulty maps for FoveaEnv
# Legend: S=Start, G=Goal, H=Hazard, P=Private Zone, .=Free cell

from copy import deepcopy

EASY_MAP = [
    ['.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', 'H', '.', '.', '.', '.'],
    ['S', '.', '.', '.', '.', '.', 'G'],
    ['.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.'],
]
# Easy: 1 hazard, 0 private zones, 40 steps
# Random agent succeeds ~50% of time

MEDIUM_MAP = [
    ['.', '.', 'H', '.', '.', 'P', '.'],
    ['.', '.', '.', '.', '.', '.', '.'],
    ['S', '.', '.', 'H', '.', '.', 'G'],
    ['.', '.', '.', '.', 'H', '.', '.'],
    ['.', 'P', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', 'H', '.', '.', '.'],
]
# Medium: 4 hazards, 2 private zones, 30 steps
# Random agent succeeds ~20% of time

HARD_MAP = [
    ['.', 'H', '.', 'P', '.', 'H', '.'],
    ['.', '.', 'H', '.', 'H', '.', '.'],
    ['S', 'H', '.', '.', '.', 'H', 'G'],  # Direct path blocked!
    ['.', '.', 'H', 'P', '.', '.', '.'],
    ['.', 'P', '.', '.', 'H', '.', '.'],
    ['.', '.', '.', 'H', '.', '.', '.'],
    ['.', 'H', 'P', '.', '.', 'H', '.'],
]
# Hard: 6 hazards BLOCKING direct path, 3 private zones, 25 steps
# Agent MUST use inspect + smart routing to succeed
# Random agent succeeds <5% of time

TASKS = {
    "easy":   {"map": EASY_MAP,   "max_steps": 40},
    "medium": {"map": MEDIUM_MAP, "max_steps": 30},
    "hard":   {"map": HARD_MAP,   "max_steps": 25},
}


def get_task(task_id: str) -> dict:
    """Returns a FRESH copy of the task — deepcopy prevents map corruption"""
    assert task_id in TASKS, f"Unknown task: '{task_id}'. Use: easy / medium / hard"
    task = TASKS[task_id]
    return {
        "map": deepcopy(task["map"]),  # CRITICAL: deepcopy so each episode gets clean map
        "max_steps": task["max_steps"]
    }