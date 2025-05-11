from collections import deque, Counter
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CauseLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # moving-window buffers for counts (last 500 infos)
        self.cause_buf     = deque(maxlen=500)
        self.stack_buf     = deque(maxlen=500)
        self.block_buf     = deque(maxlen=500)
        self.truncated_buf = deque(maxlen=500)

        # moving windows (last 500 steps)
        self.targets_buf           = deque(maxlen=500)
        self.validate_before_buf   = deque(maxlen=500)
        self.validate_after_buf    = deque(maxlen=500)
        self.steps_buf             = deque(maxlen=500)
        self.current_reward_buf    = deque(maxlen=500)
        self.current_reward        = 0.0

        # step counter
        self.num_steps = 0

    def _on_step(self) -> bool:
        self.num_steps += 1
        infos = self.locals.get("infos", [])

        for info in infos:
            # append to moving buffers for counts
            self.cause_buf.append(info.get("cause", "other"))
            self.stack_buf.append(info.get("stack", "other"))
            self.block_buf.append(info.get("block", "unknown"))
            self.truncated_buf.append(int(info.get("truncated", False)))

            # moving window metrics
            self.targets_buf.append(info.get("num_targets_reached", 0))
            self.validate_before_buf.append(info.get("validate_floor_reward_before_n_floor", 0.0))
            self.validate_after_buf.append(info.get("validate_floor_reward_after_n_floor", 0.0))
            self.steps_buf.append(info.get("steps", 0))
            cur = info.get("current_reward", 0.0)
            self.current_reward_buf.append(cur)
            self.current_reward = cur

        # record moving-window counts
        cause_counts = Counter(self.cause_buf)
        for c, cnt in cause_counts.items():
            self.logger.record(f"cause/count/{c}", cnt)
        stack_counts = Counter(self.stack_buf)
        for s, cnt in stack_counts.items():
            self.logger.record(f"stack/count/{s}", cnt)
        block_counts = Counter(self.block_buf)
        for b, cnt in block_counts.items():
            self.logger.record(f"block/count/{b}", cnt)
        self.logger.record("truncated/total", sum(self.truncated_buf))

        # helper for moving average
        def mavg(buf):
            return float(np.mean(buf)) if buf else 0.0

        # record moving window metrics
        self.logger.record("num_targets_reached",     self.targets_buf[-1] if self.targets_buf else 0)
        self.logger.record("num_targets_reached_avg", mavg(self.targets_buf))
        self.logger.record("validate_before",         self.validate_before_buf[-1] if self.validate_before_buf else 0.0)
        self.logger.record("validate_before_avg",     mavg(self.validate_before_buf))
        self.logger.record("validate_after",          self.validate_after_buf[-1] if self.validate_after_buf else 0.0)
        self.logger.record("validate_after_avg",      mavg(self.validate_after_buf))
        self.logger.record("steps",                   self.steps_buf[-1] if self.steps_buf else 0)
        self.logger.record("steps_avg",               mavg(self.steps_buf))
        self.logger.record("current_reward",          self.current_reward)
        self.logger.record("current_reward_avg",      mavg(self.current_reward_buf))

        return True
