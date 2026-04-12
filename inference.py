import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
from grader import grade_episode
from typing import TypedDict

class GradeResult(TypedDict):
    final_score: float
    navigation_score: float
    privacy_efficiency_score: float
    reached_goal: bool

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN", "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")

# ✅ Use a model that actually exists in your Ollama
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")   # or "gemma2:2b"
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b-it")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# ── OpenAI client pointing to Ollama ──────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "not-needed")

# ── Environment API wrappers ──────────────────────────────────────
def reset_environment(task_id: str = "easy"):
    try:
        response = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to reset environment: {e}")
        raise

def step_environment(move: str, look: str, inspect: bool):
    try:
        response = requests.post(
            f"{ENV_URL}/step",
            json={"move": move, "look": look, "inspect": inspect},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP {response.status_code}: {response.text}")
        raise
    except Exception as e:
        print(f"❌ Failed to step environment: {e}")
        raise

def get_state():
    try:
        response = requests.get(f"{ENV_URL}/state", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Failed to get state: {e}")
        raise

# ── LLM call with JSON cleaning ───────────────────────────────────
def call_llm(system_prompt: str, user_message: str) -> str:
    try:
        message = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=256
        )
        content = message.choices[0].message.content
        if content is None:
            return '{"move": "stay", "look": "stay", "inspect": false}'

        # Strip markdown fences
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return '{"move": "stay", "look": "stay", "inspect": false}'

# ── Main episode runner ───────────────────────────────────────────
def run_episode(task_id: str = "medium", verbose: bool = True):
    print(f"\n{'='*60}")
    print(f"Episode: {task_id.upper()}")
    print(f"{'='*60}")

    print("[START]")
    print(json.dumps({"task": task_id, "episode": 1}))
    # ✅ Single-line JSON with "type" field
    start_log = {"type": "[START]", "task": task_id, "episode": 1}
    print(json.dumps(start_log))

    obs = reset_environment(task_id)
    total_reward = 0.0
    steps = 0
    max_steps = obs.get("max_steps", 30)
    done = False

    system_prompt = """You are an AI agent navigating a grid world.
You receive a 3x3 patch of the grid as an array of strings.
Your goal is to reach the Goal ('G') while respecting private zones ('P').

You must respond with a JSON object containing three fields:
- "move": one of "up", "down", "left", "right", "stay"
- "look": one of "up", "down", "left", "right", "stay"   <-- MUST BE A DIRECTION WORD, NOT COORDINATES
- "look": one of "up", "down", "left", "right", "stay"
- "inspect": true or false

Example valid responses:
{"move": "right", "look": "right", "inspect": false}
{"move": "down", "look": "up", "inspect": true}

DO NOT output coordinates like "1,1". "look" is a direction, not a grid position.
Respond with ONLY the JSON object, no extra text or markdown."""

    while steps < max_steps:
        patch_str = json.dumps(obs.get("patch", []))
        agent_pos = obs.get("agent_pos", [0, 0])
        look_center = obs.get("look_center", [0, 0])

        user_message = f"""Current state:
- Patch (3x3 view): {patch_str}
- Agent position: {agent_pos}
- Looking at: {look_center}
- Step: {steps + 1}/{max_steps}

What is your next action? Respond with JSON only."""

        llm_response = call_llm(system_prompt, user_message)

        # Parse and validate action
        try:
            action = json.loads(llm_response)
            move = action.get("move", "stay")
            look = action.get("look", "stay")
            inspect = action.get("inspect", False)
        except (json.JSONDecodeError, TypeError):
            if verbose:
                print(f"⚠️  LLM returned invalid JSON: {llm_response}")
            move, look, inspect = "stay", "stay", False

        # 🛡️ Fallback: ensure look is a valid direction
        valid_dirs = ["up", "down", "left", "right", "stay"]
        if look not in valid_dirs:
            if verbose:
                print(f"⚠️  Invalid look direction '{look}', defaulting to 'stay'")
            look = "stay"

        # Step environment
        result = step_environment(move, look, inspect)
        obs = result
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        event = result.get("last_event", "moved")

        total_reward += reward
        steps += 1

        print("[STEP]")
        print(json.dumps({
            "step": steps, "action": {"move": move, "look": look, "inspect": inspect}, "reward": round(reward, 4),
            "done": done, "event": event
        }))
        # ✅ Single-line JSON with "type" field
        step_log = {
            "type": "[STEP]",
            "step": steps,
            "action": {"move": move, "look": look, "inspect": inspect},
            "reward": round(reward, 4),
            "done": done,
            "event": event
        }
        print(json.dumps(step_log))

        if verbose:
            print(f"Step {steps:02d} | move={move:6s} look={look:6s} inspect={inspect} | "
                  f"reward={reward:+.2f} | event={event}")

        if done:
            break

    # Final state and grading
    state = get_state()
    episode_reward = state.get("episode_reward", total_reward)
    privacy_violations = state.get("privacy_violations", 0)

    reached_goal = (obs.get("last_event") == "goal")
    score = grade_episode(
        episode_reward=episode_reward,
        reached_goal=reached_goal,
        privacy_violations=privacy_violations,
        total_steps=steps
    )

    print("[END]")
    print(json.dumps({
    "task": task_id,
    "score": round(score["final_score"], 4),
    "navigation_score": round(score["navigation_score"], 4),
    "privacy_efficiency_score": round(score["privacy_efficiency_score"], 4),
    # reached_goal removed ✅
}))
    # ✅ Single-line JSON with "type" field
    end_log = {
        "type": "[END]",
        "task": task_id,
        "score": round(score["final_score"], 4),
        "navigation_score": round(score["navigation_score"], 4),
        "privacy_efficiency_score": round(score["privacy_efficiency_score"], 4)
    }
    print(json.dumps(end_log))

    if verbose:
        print("\n" + "="*60)
        print("Episode Summary:")
        print("="*60)
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Privacy Violations: {privacy_violations}")
        print(f"  Done: {done}")
        print("="*60 + "\n")

    return episode_reward, steps, privacy_violations

def run_all_tasks():
    print("\n" + "="*60)
    print("FoveaEnv — Real Hugging Face API Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Environment: {ENV_URL}")
    print("="*60 + "\n")

    results = {}
    for task_id in ["easy", "medium", "hard"]:
        try:
            reward, steps, privacy = run_episode(task_id, verbose=True)
            results[task_id] = {"reward": reward, "steps": steps, "privacy": privacy}
        except Exception as e:
            print(f"❌ Task {task_id} failed: {e}")
            results[task_id] = {"error": str(e)}

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for task_id, data in results.items():
        if "error" in data:
            print(f"{task_id}: ❌ {data['error']}")
        else:
            print(f"{task_id}: reward={data['reward']:.2f}, steps={data['steps']}, privacy_violations={data['privacy']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_tasks()
