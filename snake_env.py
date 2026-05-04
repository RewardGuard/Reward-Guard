import os
import pygame
import random
import sys
import math
import json
import numpy as np
from enum import Enum
from collections import namedtuple, defaultdict

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

WHITE = (255, 255, 255)
BLACK = (15, 15, 20)
GRID_COLOR = (25, 25, 35)
RED = (220, 50, 50)
GREEN1 = (50, 200, 80)
GREEN2 = (40, 160, 65)

BLOCK_SIZE = 20
GAME_SPEED = 10
GRID_W = 200
GRID_H = 160

EXPECTED = {
    "survival": 0.15,
    "food": 0.60,
    "death": 0.05,
    "proximity": 0.20,
}


class SnakeEnv:

    def __init__(self, width=GRID_W, height=GRID_H, render=True):
        self.w = width
        self.h = height
        self.render_game = render
        self.reward_weights = {
            "survival": 1.0,
            "food": 1.0,
            "death": 1.0,
            "proximity": 1.0,
        }
        pygame.init()
        if self.render_game:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake RL + RewardGuard")
            self.font = pygame.font.SysFont("monospace", 14)
        else:
            self.display = pygame.display.set_mode((1, 1))
            self.font = None
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.prev_dist = self._distance_to_food()
        return self._get_state()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _distance_to_food(self):
        return math.sqrt(
            (self.head.x - self.food.x) ** 2 + (self.head.y - self.food.y) ** 2
        )

    def step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self._move(action)
        self.snake.insert(0, self.head)
        rewards = {"survival": 0.0, "food": 0.0, "death": 0.0, "proximity": 0.0}
        done = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            rewards["death"] = -50.0
        else:
            rewards["survival"] = 1.0
            if self.head == self.food:
                self.score += 1
                rewards["food"] = 10.0
                self._place_food()
            else:
                self.snake.pop()
            curr_dist = self._distance_to_food()
            rewards["proximity"] = 0.5 if curr_dist < self.prev_dist else -0.3
            self.prev_dist = curr_dist
        weighted_rewards = {
            k: v * self.reward_weights.get(k, 1.0) for k, v in rewards.items()
        }
        total_reward = sum(weighted_rewards.values())
        if self.render_game:
            self._render()
            self.clock.tick(GAME_SPEED)
        state = self._get_state()
        info = {
            "score": self.score,
            "frame": self.frame_iteration,
            "snake_length": len(self.snake),
            "dist_to_food": self._distance_to_food(),
            "total_reward": total_reward,
            "reward_components": dict(rewards),
        }
        return state, rewards, done, info

    def set_reward_weights(self, weights):
        self.reward_weights.update(weights)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action == 1:
            idx = (idx + 1) % 4
        elif action == 2:
            idx = (idx - 1) % 4
        self.direction = clock_wise[idx]
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _get_state(self):
        head = self.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        state = [
            (dir_r and self._is_collision(point_r))
            or (dir_l and self._is_collision(point_l))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l))
            or (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d)),
            (dir_d and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_u))
            or (dir_l and self._is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            self.food.x < head.x,
            self.food.x > head.x,
            self.food.y < head.y,
            self.food.y > head.y,
        ]
        return tuple(int(s) for s in state)

    def _render(self):
        self.display.fill(BLACK)
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.w, y))
        for i, pt in enumerate(self.snake):
            color = GREEN1 if i == 0 else GREEN2
            pygame.draw.rect(
                self.display, color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display,
                BLACK,
                pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4),
            )
        pygame.draw.rect(
            self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        )
        if self.font:
            score_text = self.font.render(
                f"Score: {self.score}  Steps: {self.frame_iteration}", True, WHITE
            )
            self.display.blit(score_text, (5, 5))
        pygame.display.flip()

    def close(self):
        pygame.quit()


class QLearningAgent:

    def __init__(
        self,
        n_actions=3,
        lr=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        old_q = self.q_table[state][action]
        target = (
            reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        )
        self.q_table[state][action] = old_q + self.lr * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_qtable(self, path):
        serializable = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump(serializable, f)
        print(f"Q-table saved to {path} ({len(serializable)} states)")

    def load_qtable(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for key_str, values in data.items():
            key = tuple(int(x) for x in key_str.strip("()").split(", "))
            self.q_table[key] = np.array(values)
        print(f"Q-table loaded from {path} ({len(self.q_table)} states)")
