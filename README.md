# 🚧 r-MIPRO / MIPROv2 (WIP)

> Resource-Adaptive Prompt Optimization for Multi-Agent LLM Systems
> ⚠️ This project is an early-stage **Work In Progress (WIP)** focused on building and experimenting with adaptive, resource-aware prompt optimization.

---

## 🧠 Project Vision

This repository explores a core question:

> How can we intelligently allocate optimization trials across modules or agents under real-world constraints like cost, latency, and limited budgets?

The long-term goal is to design a **resource-adaptive prompt optimizer** that:

* Dynamically identifies bottlenecks
* Allocates compute where it matters most
* Balances exploration vs exploitation
* Minimizes token cost while maximizing task performance

This is an experimental sandbox for building that system.

---

# 🧪 Current Implementations

## 🔹 MIPROv2 (Modular Iterative Prompt Optimization)

A modular optimization framework that:

* Optimizes prompt configurations per module
* Tracks convergence metrics
* Evaluates performance per trial
* Logs cost, latency, and token usage (LLM-backed runs)

Experiments currently include:

* Classification task optimization
* Generation task optimization
* Multi-module optimization (`text_extractor`, `code_generator`, `reasoner`)

This version focuses on adaptive scheduling without hard resource constraints.

---

## 🔹 r-MIPRO (Resource-Constrained Variant)

An extension of MIPROv2 that introduces:

* Cost-awareness
* Latency-awareness
* Trial budget constraints
* Agent-level resource allocation

r-MIPRO experiments currently simulate role-specialized agents:

* `product_advisor`
* `tech_specialist`
* `billing_expert`
* `escalation_handler`

The system attempts to allocate more trials to bottleneck agents under fixed budget conditions.

⚠️ Note: The constraint system is still being refined and does not yet strictly enforce latency ceilings.

---

# 📊 Experimental Status

The repository includes:

* Logged LLM-backed experiments
* JSON result files
* Multi-run benchmarking outputs
* Agent-level allocation histories

These are **research logs**, not production-ready metrics.

Performance numbers may change as:

* The scheduler is refined
* Scoring functions are improved
* Exploration strategies are redesigned

---

# ⚙️ Repository Structure

```
.
├── mipro-test3.py
├── mipro-test4-llm.py
├── r-mipro_2nd_test.py
├── *_log.txt
├── *_results.json
```

The structure will evolve as the project stabilizes.

---

# ▶️ Running Experiments

Install dependencies:

```bash
pip install -r requirements.txt
```

Run MIPROv2 module optimization:

```bash
python mipro-test3.py
```

Run LLM-backed task optimization:

```bash
python mipro-test4-llm.py
```

Run r-MIPRO agent optimization:

```bash
python r-mipro_2nd_test.py
```

---

# 🛠️ What’s Missing (Planned Work)

* Proper constraint enforcement (hard latency caps)
* Cleaner abstraction between scheduler and optimizer
* Statistical significance testing across runs
* Visualization dashboards
* Cleaner experiment configuration system
* Improved bandit-style allocation strategies

---

# 📜 Gitmoji Legend

| Gitmoji             | Meaning                      |
| ------------------- | ---------------------------- |
| 🚧 `:construction:` | Work in progress             |
| 🤖 `:robot:`        | LLM-related functionality    |
| 🧠 `:brain:`        | Optimization logic updates   |
| 🔬 `:microscope:`   | Experimental changes         |
| 📊 `:bar_chart:`    | Metrics & evaluation updates |
| 💰 `:moneybag:`     | Cost tracking updates        |
| ⏱ `:stopwatch:`     | Latency measurement updates  |
| ♻️ `:recycle:`      | Refactoring or redesign      |

---

## 👨‍💻 Author

Dhairya Bhatia
Independent Research / Systems + LLM Optimization

---

This project is evolving quickly. Expect breaking changes, experimental code, and shifting design decisions as the optimization framework matures.
