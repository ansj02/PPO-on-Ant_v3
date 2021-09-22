from model import *

def train(env, log_std, episode, T, print_interval, done_penalty, lr, no_rand_act = False):
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    model = PPO(obs_size=state_size, act_size=action_size, log_std=log_std, lr=lr)
    data_buffer = DataBuffer()
    std = np.exp(log_std)
    score = 0.
    for step in range(episode):
        state = env.reset()
        done = False
        count = 0
        while count < 600:
            for t in range(T):
                count = count+1
                act = torch.normal(model.model.pi_layer(torch.tensor(state, dtype=torch.float)), std).detach().numpy()
                if no_rand_act:
                    act = model.model.pi_layer(torch.tensor(state, dtype=torch.float)).detach().numpy()
                next_state, reward, done, info = env.step(act)
                if done:
                    reward -= done_penalty
                data_buffer.save_data(state, act, [reward], done)
                state = next_state
                score += reward
                if done:
                    break
            state_list, act_list, rew_list, done_list = data_buffer.get_data()
            data_buffer.clear_data()
            model.update(state_list=state_list, act_list=act_list, reward_list=rew_list)
            if done:
                break
        if step % print_interval == 0 and step != 0:
            print("# of episode :{}, avg score : {:.1f}, done : {}".format(step, score / print_interval, done))
            score = 0.0
    torch.save(model.model.state_dict(), 'model_data_save.pth')
    env.close()

def test(env):
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    model = PPO(obs_size=state_size, act_size=action_size, log_std=-10, lr=0)
    data_buffer = DataBuffer()

    score = 0.
    state = env.reset()
    done = False
    count = 0
    while count < 60000:
        env.render()
        count = count+1
        act = model.model.pi_layer(torch.tensor(state, dtype=torch.float)).detach().numpy()
        next_state, reward, done, info = env.step(act)
        state = next_state
        score += reward
        if done:
            break
    print("score : {:.1f}, done : {}".format(score, done))

    env.close()


