from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

# from .go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
# from .go2.go2_env import Go2Robot
from .base.legged_robot_walk import LeggedRobot
from .base.legged_robot_config_walk import Cfg, CfgPPO
from .go2.go2_config_walk import config_go2

from legged_gym.utils.task_registry import task_registry

# task_registry.register( "go2", Go2Robot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())