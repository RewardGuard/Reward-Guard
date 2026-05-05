# RewardGuard

**Trust Your AI Training**

RewardGuard is an AI alignment and safety tooling company focused on reinforcement learning systems. We provide reward auditing libraries that help developers detect reward hacking, misalignment, and training degradation early in the training process.

---

## 🎯 What is RewardGuard?

RewardGuard analyzes your RL training logs and ensures your reward functions are balanced and aligned with your intended goals. It detects when agents find unintended ways to maximize rewards (reward hacking) and provides actionable insights to fix them.

### Key Features

- **Reward Distribution Analysis** - Understand how rewards are distributed across different sources
- **Imbalance Detection** - Automatically detect when reward components are misaligned
- **Training Diagnostics** - Monitor trends and catch training issues early
- **Actionable Recommendations** - Get clear suggestions on how to fix imbalances
- **Auto-Adjustment** (Premium) - Automatically rebalance rewards during training

---

## 📦 Two Versions

### 🟢 Free Version

**What it does:**
- Analyzes reward distributions
- Detects imbalances and dominance patterns
- Provides warnings and recommendations
- Generates detailed reports

**What it doesn't do:**
- Does NOT modify training behavior
- Read-only analysis and insights

**Installation:**
```bash
pip install rewardguard
```

### 🔒 Premium Version (Private)

**Everything in Free, PLUS:**
- Automatic reward rebalancing
- Live monitoring during training
- Guardrails against reward hacking
- Continuous alignment enforcement
- Production-safe controls

**Installation:**
```bash
pip install rewardguard-premium --index-url <private-registry-url>
# Requires authentication token
```

---

## 🚀 Quick Start

### Free Version Example

```python
from rewardguard import RewardGuard

# Initialize
guard = RewardGuard(tolerance=5.0)

# Parse your training logs
episodes = guard.parse_logs(raw_log_text)

# Define expected distribution
expected = {
    "reward_a": 60.0,  # Want 60% from component A
    "reward_b": 40.0   # Want 40% from component B
}

# Analyze balance
result = guard.analyze_balance(episodes, expected)

# Print report
guard.print_analysis_report(result)
```

**Output:**
```
REWARDGUARD ANALYSIS REPORT
============================================================
📊 General Statistics:
   Episodes analyzed: 50
   Reward sources found: reward_a, reward_b

📈 Reward Distribution (%):
   Source          Real       Expected   Diff       Status
   --------------- ---------- ---------- ---------- ------------
   reward_a        75.2       60.0       +15.2      ⚠️  imbalanced
   reward_b        24.8       40.0       -15.2      ⚠️  imbalanced

🎯 Recommended Reward Weights (multipliers):
   reward_a: 0.82x (ADJUST)
   reward_b: 1.54x (ADJUST)

🔧 Summary of Actions Needed:
   • reward_a: Decrease weight by ~15.2%
   • reward_b: Increase weight by ~15.2%
```

    Here is a image of the report:
    
    <img width="1934" height="1673" alt="snake_session_report" src="https://github.com/user-attachments/assets/2dd51791-5515-47c3-8387-56dd523ca235" />


### Premium Version Example

```python
from rewardguard import AutoBalanceSystem

# Initialize with auto-tuning enabled
balance = AutoBalanceSystem(auto_tune=True)

# Define components
component_a = balance.define("component_a", initial=10.0)
component_b = balance.define("component_b", initial=5.0)

# Set expected distribution
balance.set_expected_distribution({
    "component_a": 60,
    "component_b": 40
})

# During training loop
for episode in range(100):
    # Your agent trains and collects rewards
    episode_rewards = {
        "component_a": component_a.current_value * some_calculation(),
        "component_b": component_b.current_value * some_calculation()
    }
    
    # Log performance - RewardGuard auto-adjusts every 10 episodes
    balance.log_performance({
        "rewards": episode_rewards,
        "outcome": "success",
        "steps": 100,
        "score": sum(episode_rewards.values())
    })

# Get final adjusted values
final_values = balance.get_current_values()
print(f"Auto-adjusted values: {final_values}")
```

---

## 📖 Use Cases

### 1. Game AI
Ensure your game AI learns to play properly, not exploit bugs:
- Detect when agents farm easy points instead of completing objectives
- Balance combat vs exploration rewards
- Prevent exploit-based strategies

### 2. Robotics
Keep robots aligned with safety and task completion:
- Balance speed vs safety rewards
- Ensure proper task prioritization
- Detect reward shortcuts

### 3. Recommendation Systems
Align recommendation rewards with business goals:
- Balance engagement vs revenue
- Prevent clickbait optimization
- Ensure long-term user satisfaction

### 4. General RL Research
Debug and optimize any RL training:
- Understand reward dynamics
- Catch training issues early
- Validate reward function design

---

## 🏗️ How It Works

### Free Version (Analysis Only)

1. **Parse Logs** - Extracts reward data from training logs
2. **Aggregate** - Calculates actual reward distribution
3. **Compare** - Compares against your expected distribution
4. **Recommend** - Suggests specific weight adjustments

**Key Principle:** Tells you what's wrong, you fix it manually.

### Premium Version (Auto-Fix)

1. **All Free features**, PLUS:
2. **Monitor** - Tracks performance over time
3. **Detect** - Identifies imbalances automatically
4. **Adjust** - Modifies reward weights in real-time
5. **Learn** - Continuously tunes based on results

**Key Principle:** Fixes problems for you automatically.

---

## 🎓 Philosophy

We believe AI should be:
- **Transparent** - You should understand what your AI is learning
- **Aligned** - Reward functions should incentivize intended behaviors
- **Safe** - Training should be monitored for unintended outcomes

RewardGuard helps ensure your models learn what you intend, not just how to maximize scores.

---

## 💰 Pricing

| Feature | Free | Premium |
|---------|------|---------|
| Reward analysis | ✅ | ✅ |
| Imbalance detection | ✅ | ✅ |
| Recommendations | ✅ | ✅ |
| Auto-adjustment | ❌ | ✅ |
| Live monitoring | ❌ | ✅ |
| Unlimited training steps | ❌ | ✅ |
| Priority support | ❌ | ✅ |
| **Price** | **$0** | **From $20 (credits)** |

---

## 📚 Documentation

- **Website:** [rewardguard.dev]
- **Tutorials:** https://youtu.be/ySif89GQ3N4
- **Docs:** [https://rewardguard.dev/docs](https://rewardguard.dev/docs)

---

## 🤝 Support

- **Community (Free):** https://www.youtube.com/@RewardGuard


---

## 📄 License

- **Free Version:** MIT License
- **Premium Version:** Proprietary

---

## 🚧 Roadmap

- [ ] Support for more log formats
- [ ] Built-in visualization dashboard
- [ ] Integration with popular RL frameworks (Stable-Baselines3, RLlib)
- [ ] Cloud-based monitoring
- [ ] Team collaboration features
- [ ] Custom alerting rules

---

## ⚡ Quick Links

- [Get Started Free](#-quick-start)
- [Upgrade to Premium](#-pricing)
- [View Examples](rewardguard.dev/demo)
- [Read the Docs](rewardguard.dev/docs)

---

**RewardGuard © 2026 | Trust Your AI**
