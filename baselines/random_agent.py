import json
import random
import requests

BASE_URL = "http://127.0.0.1:7860"
TASKS = ["easy", "medium", "hard"]

MOVES = ["up", "down", "left", "right", "stay"]
LOOKS = ["up", "down", "left", "right", "stay"]


def random_action():
    return {
        "move": random.choice(MOVES),
        "look": random.choice(LOOKS),
        "inspect": random.choice([True, False]),
    }


def run_episode(task_id: str):
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    if r.status_code != 200:
        print(f"❌ Reset failed for {task_id}: {r.status_code} {r.text[:200]}")
        return None

    obs = r.json()
    total_reward = 0.0

    print(json.dumps({
        "type": "[START]",
        "task": task_id,
        "episode": 1,
        "max_steps": obs["max_steps"],
        "start_pos": obs["agent_pos"]
    }))

    while True:
        action = random_action()

        resp = requests.post(f"{BASE_URL}/step", json=action)
        if resp.status_code != 200:
            print(f"⚠️ Server error {resp.status_code}: {resp.text[:200]}")
            return None

        obs = resp.json()
        reward = obs.get("reward", 0.0)
        done = obs.get("done", False)
        total_reward += reward

        print(json.dumps({
            "type": "[STEP]",
            "task": task_id,
            "step": obs.get("step_count", 0),
            "action": action,
            "reward": reward,
            "done": done,
            "event": obs.get("last_event"),
            "agent_pos": obs.get("agent_pos")
        }))

        if done:
            score = obs.get("score") or obs
            end_log = {
                "type": "[END]",
                "task": task_id,
                "episode": 1,
                "total_reward": round(total_reward, 4),
                "navigation_score": score.get("navigation_score", 0.5),
                "privacy_efficiency_score": score.get("privacy_efficiency_score", 0.5),
                "final_score": score.get("final_score", 0.5),
                "reached_goal": score.get("reached_goal", False),
            }
            print(json.dumps(end_log))
            return end_log


def run_all():
    results = {}
    for task in TASKS:
        results[task] = run_episode(task)

    print("\n📊 RANDOM AGENT SUMMARY")
    print(f"{'Task':<10} {'Final Score':<12} {'Reached Goal':<14}")
    print("-" * 40)

    for task, result in results.items():
        if result:
            print(f"{task:<10} {result['final_score']:<12.4f} {str(result['reached_goal']):<14}")
        else:
            print(f"{task:<10} {'ERROR':<12} {'ERROR':<14}")


if __name__ == "__main__":
    run_all()
