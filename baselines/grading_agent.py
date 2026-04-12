import json
import requests

BASE_URL = "http://127.0.0.1:7860"
TASKS = ["easy", "medium", "hard"]


def find_goal(grid):
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "G":
                return r, c
    return None


def run_episode(task_id: str):
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    if r.status_code != 200:
        return {"task": task_id, "error": f"reset failed: {r.status_code} {r.text[:200]}"}

    obs = r.json()
    total_reward = 0.0

    print(json.dumps({
        "type": "[START]",
        "task": task_id,
        "episode": 1,
        "max_steps": obs["max_steps"],
        "start_pos": obs["agent_pos"]
    }))

    state_resp = requests.get(f"{BASE_URL}/state")
    if state_resp.status_code != 200:
        return {"task": task_id, "error": f"state failed: {state_resp.status_code} {state_resp.text[:200]}"}

    full_state = state_resp.json()
    goal = find_goal(full_state["full_grid"])
    if goal is None:
        return {"task": task_id, "error": "goal not found"}

    goal_row, goal_col = goal

    while True:
        row, col = obs["agent_pos"]

        if col < goal_col:
            move = "right"
        elif col > goal_col:
            move = "left"
        elif row < goal_row:
            move = "down"
        elif row > goal_row:
            move = "up"
        else:
            move = "stay"

        action = {
            "move": move,
            "look": "stay",
            "inspect": False
        }

        resp = requests.post(f"{BASE_URL}/step", json=action)
        if resp.status_code != 200:
            return {"task": task_id, "error": f"step failed: {resp.status_code} {resp.text[:200]}"}

        obs = resp.json()
        total_reward += obs.get("reward", 0.0)

        print(json.dumps({
            "type": "[STEP]",
            "task": task_id,
            "step": obs.get("step_count", 0),
            "action": action,
            "reward": obs.get("reward", 0.0),
            "done": obs.get("done", False),
            "event": obs.get("last_event"),
            "agent_pos": obs.get("agent_pos")
        }))

        if obs.get("done", False):
            score = obs.get("score") or obs
            result = {
                "type": "[END]",
                "task": task_id,
                "episode": 1,
                "total_reward": round(total_reward, 4),
                "navigation_score": score.get("navigation_score", 0.5),
                "privacy_efficiency_score": score.get("privacy_efficiency_score", 0.5),
                "final_score": score.get("final_score", 0.5),
                "reached_goal": score.get("reached_goal", False),
            }
            print(json.dumps(result))
            return result


def run_all():
    results = {}
    for task in TASKS:
        results[task] = run_episode(task)

    print("\n📊 GRADING AGENT SUMMARY")
    print(f"{'Task':<10} {'Final Score':<12} {'Reached Goal':<14}")
    print("-" * 40)

    for task, result in results.items():
        if result and "final_score" in result:
            print(f"{task:<10} {result['final_score']:<12.4f} {str(result['reached_goal']):<14}")
        else:
            print(f"{task:<10} {'ERROR':<12} {'ERROR':<14}")


if __name__ == "__main__":
    run_all()
