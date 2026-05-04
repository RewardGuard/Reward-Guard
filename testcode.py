"""
Tests for Snake RL + RewardGuard integration (testcode.py).
Run with: pytest test_snake.py -v
"""

import os
import json
import math
import random
import pytest
import numpy as np
from collections import defaultdict
from unittest.mock import patch, MagicMock

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# ── Import the modules under test ──────────────────────────────────────────────
from testcode import (
    SnakeEnv,
    QLearningAgent,
    Direction,
    Point,
    BLOCK_SIZE,
    GRID_W,
    GRID_H,
    EXPECTED,
)
import rewardguard as rg


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def env():
    e = SnakeEnv(render=False)
    yield e
    e.close()


@pytest.fixture
def agent():
    return QLearningAgent()


@pytest.fixture
def monitor():
    return rg.Monitor(expected=EXPECTED, tolerance=5.0, window=200, max_history=10_000)


# ══════════════════════════════════════════════════════════════════════════════
# SnakeEnv — reset & state
# ══════════════════════════════════════════════════════════════════════════════

class TestSnakeEnvReset:

    def test_reset_returns_tuple_of_ints(self, env):
        state = env.reset()
        assert isinstance(state, tuple)
        assert all(v in (0, 1) for v in state), "State must be binary"

    def test_reset_state_length(self, env):
        state = env.reset()
        assert len(state) == 11, "State must have 11 features"

    def test_reset_score_is_zero(self, env):
        env.reset()
        assert env.score == 0

    def test_reset_snake_has_three_segments(self, env):
        env.reset()
        assert len(env.snake) == 3

    def test_reset_head_centered(self, env):
        env.reset()
        assert env.head == Point(GRID_W // 2, GRID_H // 2)

    def test_reset_direction_right(self, env):
        env.reset()
        assert env.direction == Direction.RIGHT

    def test_food_not_on_snake(self, env):
        env.reset()
        assert env.food not in env.snake

    def test_food_within_bounds(self, env):
        for _ in range(20):
            env.reset()
            assert 0 <= env.food.x < GRID_W
            assert 0 <= env.food.y < GRID_H


# ══════════════════════════════════════════════════════════════════════════════
# SnakeEnv — step mechanics
# ══════════════════════════════════════════════════════════════════════════════

class TestSnakeEnvStep:

    def test_step_returns_four_values(self, env):
        env.reset()
        result = env.step(0)
        assert len(result) == 4

    def test_step_info_keys(self, env):
        env.reset()
        _, _, _, info = env.step(0)
        for key in ("score", "frame", "snake_length", "dist_to_food", "total_reward", "reward_components"):
            assert key in info

    def test_reward_components_present(self, env):
        env.reset()
        _, rewards, _, _ = env.step(0)
        for key in ("survival", "food", "death", "proximity"):
            assert key in rewards

    def test_survival_reward_when_alive(self, env):
        env.reset()
        _, rewards, done, _ = env.step(0)
        if not done:
            assert rewards["survival"] == 1.0

    def test_death_reward_on_collision(self, env):
        """Force a wall collision and verify death reward."""
        env.reset()
        env.direction = Direction.LEFT
        env.head = Point(0, GRID_H // 2)
        env.snake = [env.head]
        _, rewards, done, _ = env.step(0)  # straight → hits wall
        assert done
        assert rewards["death"] == -50.0

    def test_frame_iteration_increments(self, env):
        env.reset()
        env.step(0)
        assert env.frame_iteration == 1
        env.step(0)
        assert env.frame_iteration == 2

    def test_timeout_triggers_done(self, env):
        env.reset()
        env.frame_iteration = 100 * len(env.snake)
        _, _, done, _ = env.step(0)
        assert done

    def test_set_reward_weights(self, env):
        env.reset()
        env.set_reward_weights({"food": 5.0})
        assert env.reward_weights["food"] == 5.0

    def test_weighted_total_reward(self, env):
        """total_reward must equal sum of weighted components."""
        env.reset()
        env.set_reward_weights({"survival": 2.0, "food": 3.0, "death": 1.0, "proximity": 1.0})
        _, rewards, done, info = env.step(0)
        expected_total = sum(
            rewards[k] * env.reward_weights.get(k, 1.0) for k in rewards
        )
        assert abs(info["total_reward"] - expected_total) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# SnakeEnv — distance helper
# ══════════════════════════════════════════════════════════════════════════════

class TestDistanceToFood:

    def test_distance_is_nonnegative(self, env):
        env.reset()
        assert env._distance_to_food() >= 0

    def test_distance_zero_when_on_food(self, env):
        env.reset()
        env.food = env.head
        assert env._distance_to_food() == 0.0

    def test_distance_formula(self, env):
        env.reset()
        env.head = Point(0, 0)
        env.food = Point(30, 40)
        assert math.isclose(env._distance_to_food(), 50.0)


# ══════════════════════════════════════════════════════════════════════════════
# QLearningAgent
# ══════════════════════════════════════════════════════════════════════════════

class TestQLearningAgent:

    def test_act_returns_valid_action(self, agent):
        state = (0,) * 11
        for _ in range(50):
            a = agent.act(state)
            assert a in (0, 1, 2)

    def test_epsilon_one_always_random(self):
        ag = QLearningAgent(epsilon=1.0)
        state = (0,) * 11
        actions = {ag.act(state) for _ in range(100)}
        assert len(actions) > 1, "With epsilon=1 should explore"

    def test_epsilon_zero_greedy(self):
        ag = QLearningAgent(epsilon=0.0)
        state = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        ag.q_table[state] = np.array([0.1, 5.0, 0.3])
        assert ag.act(state) == 1

    def test_learn_updates_q_value(self, agent):
        s = (0,) * 11
        s2 = (1,) * 11
        old_q = agent.q_table[s][0]
        agent.learn(s, 0, 1.0, s2, False)
        assert agent.q_table[s][0] != old_q

    def test_learn_done_ignores_next_state(self, agent):
        s = (0,) * 11
        s2 = (1,) * 11
        agent.q_table[s2] = np.array([999.0, 999.0, 999.0])
        agent.learn(s, 0, -50.0, s2, done=True)
        # target = reward only, so q shouldn't approach 999
        assert agent.q_table[s][0] < 100

    def test_epsilon_decay(self, agent):
        original = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < original

    def test_epsilon_never_below_min(self):
        ag = QLearningAgent(epsilon=0.011, epsilon_decay=0.5, epsilon_min=0.01)
        for _ in range(20):
            ag.decay_epsilon()
        assert ag.epsilon >= 0.01

    def test_save_and_load_qtable(self, agent, tmp_path):
        state = (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
        agent.q_table[state] = np.array([1.0, 2.0, 3.0])
        path = str(tmp_path / "qtable.json")
        agent.save_qtable(path)
        new_agent = QLearningAgent()
        new_agent.load_qtable(path)
        np.testing.assert_array_almost_equal(new_agent.q_table[state], [1.0, 2.0, 3.0])

    def test_save_qtable_creates_file(self, agent, tmp_path):
        path = str(tmp_path / "qtable.json")
        agent.save_qtable(path)
        assert os.path.exists(path)


# ══════════════════════════════════════════════════════════════════════════════
# RewardGuard Monitor integration
# ══════════════════════════════════════════════════════════════════════════════

class TestRewardGuardMonitor:

    def test_monitor_initial_step_count(self, monitor):
        assert monitor.step_count == 0

    def test_monitor_expected_matches_constant(self, monitor):
        assert monitor.expected == EXPECTED

    def test_step_increments_count(self, monitor):
        monitor.step({"survival": 1.0, "food": 0.0, "death": 0.0, "proximity": 0.5})
        assert monitor.step_count == 1

    def test_check_returns_result(self, monitor):
        for _ in range(10):
            monitor.step({"survival": 1.0, "food": 0.0, "death": 0.0, "proximity": 0.5},
                         episode_done=True)
        result = monitor.check()
        assert result is not None

    def test_result_has_severity(self, monitor):
        for _ in range(10):
            monitor.step({"survival": 1.0, "food": 0.0, "death": 0.0, "proximity": 0.5},
                         episode_done=True)
        result = monitor.check()
        assert hasattr(result, "severity")

    def test_result_has_real_percentages(self, monitor):
        for _ in range(20):
            monitor.step({"survival": 1.0, "food": 10.0, "death": 0.0, "proximity": 0.5},
                         episode_done=True)
        result = monitor.check()
        assert hasattr(result, "real_percentages")
        assert abs(sum(result.real_percentages.values()) - 100.0) < 1.0

    def test_reset_clears_step_count(self, monitor):
        monitor.step({"survival": 1.0, "food": 0.0, "death": 0.0, "proximity": 0.0})
        monitor.reset()
        assert monitor.step_count == 0

    def test_expected_percentages_sum_to_100(self):
        total = sum(EXPECTED.values())
        assert abs(total - 1.0) < 1e-9 or abs(total - 100.0) < 1e-9, \
            "EXPECTED values must sum to 1.0 or 100.0"


# ══════════════════════════════════════════════════════════════════════════════
# Full episode integration smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestFullEpisodeIntegration:

    def test_single_episode_runs_without_error(self):
        env = SnakeEnv(render=False)
        agent = QLearningAgent()
        monitor = rg.Monitor(expected=EXPECTED, tolerance=5.0, window=200)

        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = agent.act(state)
            next_state, rewards, done, info = env.step(action)
            agent.learn(state, action, info["total_reward"], next_state, done)
            monitor.step(
                {k: rewards[k] for k in ("survival", "food", "death", "proximity")},
                episode_done=done,
            )
            state = next_state
            steps += 1

        assert steps > 0
        assert monitor.step_count == steps
        env.close()

    def test_multiple_episodes_accumulate_steps(self):
        env = SnakeEnv(render=False)
        agent = QLearningAgent()
        monitor = rg.Monitor(expected=EXPECTED, tolerance=5.0, window=200)

        total_steps = 0
        for _ in range(3):
            state = env.reset()
            done = False
            ep_steps = 0
            while not done and ep_steps < 200:
                action = agent.act(state)
                next_state, rewards, done, info = env.step(action)
                agent.learn(state, action, info["total_reward"], next_state, done)
                monitor.step(
                    {k: rewards[k] for k in ("survival", "food", "death", "proximity")},
                    episode_done=done,
                )
                state = next_state
                ep_steps += 1
            total_steps += ep_steps

        assert monitor.step_count == total_steps
        env.close()
