# test_basic.py
# Run this after env.py and grader.py are complete to verify everything works
# Command: python test_basic.py

from env import FoveaEnv
from models import BlinkAction
from grader import grade_episode

print("=" * 50)
print("FoveaEnv — Full System Test")
print("=" * 50)

env = FoveaEnv()

# ── Test 1: reset() ──────────────────────────────
obs = env.reset("easy")
assert obs.step_count == 0, "step_count should be 0 after reset"
assert obs.agent_pos == [2, 0], f"Agent should start at [2,0], got {obs.agent_pos}"
assert len(obs.patch) == 3, "Patch should be 3 rows"
assert len(obs.patch[0]) == 3, "Patch should be 3 cols"
assert obs.last_event == "start", "First event should be 'start'"
print("✅ Test 1 PASSED — reset() works correctly")

# ── Test 2: state() right after reset ────────────
state0 = env.state()
assert isinstance(state0.episode_id, str), "episode_id must be a string"
assert state0.episode_id.strip() != "", "episode_id must not be empty"
assert len(state0.full_grid) == 7, "Grid should have 7 rows"
assert len(state0.full_grid[0]) == 7, "Grid should have 7 cols"
assert state0.step_count == 0, "step_count should be 0 after reset"
assert state0.done is False, "Episode should not be done after reset"
assert state0.privacy_violations == 0, "privacy_violations should start at 0"
print("✅ Test 2 PASSED — state() returns full episode metadata")

# ── Test 3: step() move ──────────────────────────
action = BlinkAction(move="right", look="stay", inspect=False)
obs2, reward, done = env.step(action)
assert obs2.step_count == 1, "step_count should be 1"
assert isinstance(reward, float), "reward must be float"
assert isinstance(done, bool), "done must be bool"
assert obs2.agent_pos == [2, 1], f"Agent should be at [2,1], got {obs2.agent_pos}"
assert obs2.last_event == "moved", f"Expected 'moved', got {obs2.last_event}"
print("✅ Test 3 PASSED — step() move works")

# ── Test 4: Hazard penalty ───────────────────────
env.reset("easy")
env.agent_pos = [1, 1]  # place agent next to hazard at [1,2]
action_into_hazard = BlinkAction(move="right", look="stay", inspect=False)
obs3, reward3, done3 = env.step(action_into_hazard)
assert reward3 <= -0.5, f"Hazard hit should give <= -0.5, got {reward3}"
assert obs3.last_event == "hazard_hit", f"Expected hazard_hit, got {obs3.last_event}"
print("✅ Test 4 PASSED — hazard penalty works")

# ── Test 5: Goal reached ─────────────────────────
env.reset("easy")
env.agent_pos = [2, 5]  # one step before goal [2,6]
action_to_goal = BlinkAction(move="right", look="stay", inspect=False)
obs4, reward4, done4 = env.step(action_to_goal)
assert done4 is True, "Episode should be done after reaching goal"
assert reward4 >= 0.9, f"Goal reward should be >= 0.9 (after step penalty), got {reward4}"
assert obs4.last_event == "goal", f"Expected goal, got {obs4.last_event}"
print("✅ Test 5 PASSED — goal reached correctly")

# ── Test 6: Timeout ──────────────────────────────
env.reset("easy")
env.max_steps = 2
env.step(BlinkAction(move="stay", look="stay", inspect=False))
_, _, done_timeout = env.step(BlinkAction(move="stay", look="stay", inspect=False))
assert done_timeout is True, "Should be done after max_steps"
print("✅ Test 6 PASSED — timeout works")

# ── Test 7: Grader strict range ──────────────────
for ep_r, goal, priv, steps in [
    (-5.0, False, 30, 25),
    (2.0, True, 0, 10),
    (0.0, False, 0, 1),
]:
    score = grade_episode(ep_r, goal, priv, steps)
    assert 0.0 < score["final_score"] < 1.0, f"final_score out of range: {score}"
    assert 0.0 < score["navigation_score"] < 1.0, f"navigation_score out of range: {score}"
    assert 0.0 < score["privacy_efficiency_score"] < 1.0, f"privacy_efficiency_score out of range: {score}"

print("✅ Test 7 PASSED — all grader scores are clamped to 0.0001..0.9999")

# ── Test 8: All 3 tasks reset ────────────────────
episode_ids = set()
for task_id in ["easy", "medium", "hard"]:
    obs = env.reset(task_id)
    state = env.state()
    assert obs.step_count == 0
    assert obs.last_event == "start"
    assert isinstance(state.episode_id, str) and state.episode_id.strip() != ""
    episode_ids.add(state.episode_id)

assert len(episode_ids) == 3, "Each reset should generate a new episode_id"
print("✅ Test 8 PASSED — all 3 tasks reset correctly")

print()
print("=" * 50)
print("🏆 ALL 8 TESTS PASSED — Ready to build server!")
print("=" * 50)
