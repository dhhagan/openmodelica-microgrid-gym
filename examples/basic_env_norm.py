import gym

if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv_test-v1',
                   net='../net/net_singleinverter.yaml',
                   model_path='../fmu/grid.network.fmu',
                   is_normalized=True)

    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action
    env.close()
