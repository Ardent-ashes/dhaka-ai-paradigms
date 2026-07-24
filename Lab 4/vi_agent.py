import time


class ValueIterationAgent:
    def __init__(self, env, epsilon=1.0, max_iter=1000, verbose=True):
        self.env = env
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.verbose = verbose
        self.U = {}
        self.policy = {}
        self.history = []          # delta per sweep (convergence curve)
        self.rewards = []          # greedy policy reward per sweep (like QL curve)

    def _q(self, s, a, U):
        """Expected value of taking action a in state s: sum T*(r + gamma*U)."""
        g = self.env.gamma
        return sum(p * (r + g * U.get(sp, 0.0))
                   for p, sp, r in self.env.transitions(s, a))

    def _greedy_reward(self, states, U):
        """Evaluate the current greedy policy's reward (fast deterministic rollout)."""
        pi = {s: max(self.env.actions(s), key=lambda a: self._q(s, a, U))
              for s in states}
        return self.env.evaluate_policy(pi)

    def solve(self):
        env = self.env
        t0 = time.perf_counter()
        states = [s for s in env.states() if not env.is_terminal(s)]
        U = {s: 0.0 for s in states}
        threshold = self.epsilon * (1 - env.gamma) / env.gamma

        for it in range(1, self.max_iter + 1):
            delta = 0.0
            for s in states:
                best = max(self._q(s, a, U) for a in env.actions(s))
                delta = max(delta, abs(best - U[s]))
                U[s] = best
            self.history.append(delta)
            self.rewards.append(self._greedy_reward(states, U))   # ← reward curve
            if self.verbose and (it % 10 == 0 or delta < threshold):
                print(f"        [VI] iter {it:>3}  delta={delta:.4f}  "
                      f"reward={self.rewards[-1]:.0f}")
            if delta < threshold:
                break

        policy = {}
        for s in states:
            policy[s] = max(env.actions(s), key=lambda a: self._q(s, a, U)) # policy
        self.U, self.policy = U, policy
        self.runtime = time.perf_counter() - t0
        self.iters = len(self.history)
        if self.verbose:
            print(f"        [VI] converged in {self.iters} iters, "
                  f"{self.runtime:.2f}s")
        return policy

