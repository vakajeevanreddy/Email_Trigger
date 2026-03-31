import random
from env.models import Observation, Action, Reward
from env.tasks import TASKS
from env.grader import grade_step

class EmailEnv:
    def __init__(self):
        self.current_task = None
        self.state_data = {}
        self.step_count = 0

    def reset(self):
        self.current_task = random.choice(TASKS)
        self.step_count = 0

        self.state_data = {
            "email": self.current_task["email"],
            "done": False
        }

        return Observation(
            email_id=self.current_task["id"],
            email_text=self.current_task["email"],
            step_count=self.step_count
        )

    def step(self, action: Action):
        self.step_count += 1

        reward_value, done, info = grade_step(
            self.current_task,
            action,
            self.step_count
        )

        reward = Reward(value=reward_value, reason=info)

        observation = Observation(
            email_id=self.current_task["id"],
            email_text=self.current_task["email"],
            step_count=self.step_count
        )

        return observation, reward, done, info

    def state(self):
        return self.state_data