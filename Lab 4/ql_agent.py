import time
from collections import defaultdict
import numpy as np


class QLearningAgent:
    def __init__(self, env, alpha=0.1, episodes=3000, max_steps=200,
                 eps_start=1.0, eps_end=0.05, seed=42, verbose=True,
                 conv_window=100, conv_tol=50.0, conv_patience=3):
        self.env = env
        self.alpha = alpha
        self.gamma = env.gamma
        self.episodes = episodes
        self.max_steps = max_steps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.conv_window   = conv_window    # rolling window size
        self.conv_tol      = conv_tol       # max change between windows to declare convergence
        self.conv_patience = conv_patience  # how many consecutive stable windows before stopping
        self.Q = defaultdict(dict)          # Q[s][a]
        self.rewards = []                   # total reward per episode (curve)
        self.conv_ep = None                 # episode at which convergence was detected
        self.policy = {}

    def _qmax(self, s):
        q = self.Q.get(s)
        return max(q.values()) if q else 0.0

    def _best(self, s):
        q = self.Q[s]
        for a in self.env.actions(s):       # ensure every action seen
            q.setdefault(a, 0.0)
        return max(q, key=q.get)

    def train(self):
        env = self.env
        t0 = time.perf_counter()
        decay = (self.eps_end / self.eps_start) ** (1.0 / max(1, self.episodes))
        eps = self.eps_start
        stable_count = 0   # consecutive stable windows counter
        prev_avg = None    # previous window's average reward

        for ep in range(1, self.episodes + 1):
            s = env.reset()
            total = 0.0
            for _ in range(self.max_steps):
                acts = env.actions(s)
                if self.rng.random() < eps:                 # explore
                    a = acts[self.rng.integers(len(acts))]
                else:                                        # exploit
                    a = self._best(s)
                sp, r, done = env.step(s, a)
                self.Q[s].setdefault(a, 0.0)
                target = r + self.gamma * self._qmax(sp)
                self.Q[s][a] += self.alpha * (target - self.Q[s][a])
                total += r
                s = sp
                if done:
                    break
            self.rewards.append(total)
            eps = max(self.eps_end, eps * decay)

            # --- convergence check (every conv_window episodes) ---
            if ep >= self.conv_window * 2 and ep % self.conv_window == 0:
                curr_avg = np.mean(self.rewards[-self.conv_window:])
                if prev_avg is not None:
                    change = abs(curr_avg - prev_avg)
                    print(f"        [CONV] ep={ep}  curr_avg={curr_avg:.1f}  change={change:.2f}  stable_count={stable_count}  tol={self.conv_tol}")
                    if change < self.conv_tol:
                        stable_count += 1
                        if stable_count >= self.conv_patience:
                            self.conv_ep = ep
                            if self.verbose:
                                print(f"        [QL] converged at episode {ep} "
                                      f"(avg reward change={change:.2f} < tol={self.conv_tol})")
                            break
                    else:
                        stable_count = 0  # reset if unstable again
                prev_avg = curr_avg

            if self.verbose and ep % max(1, self.episodes // 10) == 0:
                avg = np.mean(self.rewards[-self.conv_window:])
                print(f"        [QL] ep {ep:>4}/{self.episodes}  "
                      f"eps={eps:.3f}  avg{self.conv_window}={avg:8.1f}")

        # greedy policy from the learned Q-table
        self.policy = {s: self._best(s) for s in self.Q}
        self.iters  = len(self.rewards)   # actual episodes run
        self.runtime = time.perf_counter() - t0
        if self.verbose:
            status = (f"converged at ep {self.conv_ep}"
                      if self.conv_ep else f"ran all {self.episodes} episodes")
            print(f"        [QL] {status}, runtime={self.runtime:.2f}s")
        return self.policy
