from tetra import *

if __name__ == '__main__':
    mode = 'test'
    env = gym.make('Ant-v3')
    random_seed = 9702

    torch.manual_seed(random_seed)
    if mode == 'train':
        num_step = 3
        fir_log_std = [-0.9, -3, -10]  # -0.9 -> -10
        log_std = [-3, -5, -10]
        episode = [5000, 5000, 5000]
        T = [20, 40, 60]
        print_interval = 20
        done_penalty = [100, 200, 300]  # 100 -> 300
        lr = [1e-4, 5e-5, 1e-5]  # 1e-4 -> 1e-5
        for idx in range(num_step):
            train(env, log_std[idx], episode[idx], T[idx], print_interval, done_penalty[idx], lr[idx])

    elif mode == 'test':
        test(env)