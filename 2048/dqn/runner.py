from env import Env2048

env = Env2048()
observation = env.reset()
lookup = {"w": 0, "s": 1, "a": 2, "d": 3}
for t in range(1000000):
    env.render()
    action = lookup[input("Move : ")]
    observation, reward, done, info = env.step(action)
    print(f"Reward: {reward}")
    if done:
        env.render()
        print("Episode finished after {} timesteps".format(t + 1))
        break
env.close()
