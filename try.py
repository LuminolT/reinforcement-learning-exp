from modelfree.model_free import ModelFree
import numpy as np


class Sarsa(ModelFree):
    def __init__(self, env_name):
        super(Sarsa, self).__init__(env_name)

    def run(self):
        q_table = np.zeros((self.n_states, self.n_states, 3))
        for i in range(self.max_iteration):
            obs = self.env.reset()
            total_reward = 0
            lr = max(self.min_lr, self.init_lr * (0.85 ** (i // 100)))
            # each iteration
            for j in range(self.max_time):
                position, velocity = self.obs_to_state(obs)
                action = self.epsilon_greedy(q_table, position, velocity)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                # update q_table
                position_, velocity_ = self.obs_to_state(obs)
                action_ = self.epsilon_greedy(q_table, position_, velocity_)
                q_table[position][velocity][action] = q_table[position][velocity][action] + lr * (
                            reward + self.gamma * q_table[position_][velocity_][action_] - q_table[position][velocity][
                        action])
                if done:
                    break
            if not i % 200:
                print(f'[{i + 1} iteration]:total_reward:{total_reward}')
        np.save("Sarsa_q_table", q_table)
        solution_policy = np.argmax(q_table, axis=2)
        solution_policy_scores = [self.run_episode(solution_policy) for _ in range(100)]
        print(f'Average score of solution :{np.mean(solution_policy_scores)}')
        self.env.closs()


if __name__ == '__main__':
    env_name = "MountainCar-v0"
    sarsa = Sarsa(env_name)
    sarsa.run()