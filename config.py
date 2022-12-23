# common
eps = 1e-9

# dplan
batch_size = 32
num_steps_per_iteration = 2000
log_interval = 1
num_test_trajectories = 1
iter = 30
init_epsilon = 1
final_epsilon = 0.1
save_model_interval = 1
start_timestep = 10000
buffer = 100000
device = 'cpu'
update_target_network_interval = 10000
gamma = 0.99
tau = 0.5
n = 1
learning_rate = 0.00025
momentum = 0.95
hidden_dims = [20]

# environment
sample_num = 1000
p = 0.5
