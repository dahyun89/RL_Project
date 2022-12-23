import loader
from dplan import DQNAgent
from dplan import DQNTrainer
import click
from common import Logger
from common import set_device_and_logger, set_global_seed
from dplan import ReplayBuffer
from environment import Environment

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.option("--log-dir", default="logs")
@click.option("--gpu", type=int, default=0)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=3)
@click.option("--info", type=str, default="")

def main(log_dir, gpu, print_log, seed, info):
    set_global_seed(seed)

    env_name = "DPLAN"
    logging = Logger(log_dir, prefix=env_name + "-" + info, print_to_terminal=print_log)
    logging.log_stamp("logging to {}".format(logging.log_path))
    dev = set_device_and_logger(gpu, logging)

    logging.log_stamp("Data Load")
    anomaly_data, unlabeld_data, test_data, test_label = loader.load_ann()

    logging.log_stamp("Set Environment")
    env = Environment(anomaly_data, unlabeld_data, test_data, test_label)
    eval_env = Environment(anomaly_data, unlabeld_data, test_data, test_label)

    logging.log_stamp("Set Buffer")
    buffer = ReplayBuffer(env.obs_dim)

    logging.log_stamp("Set Agent")
    agent = DQNAgent(env.obs_dim, env.action_dim)
    env.refresh_net(agent.q_network)
    eval_env.refresh_net(agent.q_network)
    env.refresh_iforest(agent.q_network)
    eval_env.refresh_iforest(agent.q_network)

    logging.log_stamp("Set Trainer")
    trainer = DQNTrainer(agent, env, eval_env, buffer, logging)

    logging.log_stamp("Starting Training")
    trainer.train()


if __name__ == '__main__':
    main()
