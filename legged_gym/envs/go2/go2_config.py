from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class env(LeggedRobotCfg.env):
        num_observations = 45
        # num_privileged_obs = 45
        num_actions = 12

    # class domain_rand(LeggedRobotCfg.domain_rand):
    #     randomize_friction = True
    #     friction_range = [0.1, 1.25]
    #     randomize_base_mass = True
    #     added_mass_range = [-1., 3.]
    #     push_robots = True
    #     push_interval_s = 5
    #     max_push_vel_xy = 1.5

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            # alive=0.15

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    # Use LSTM for standing task
    # class policy:
    #     init_noise_std = 0.8
    #     actor_hidden_dims = [32]
    #     critic_hidden_dims = [32]
    #     activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    #     # only for 'ActorCriticRecurrent':
    #     rnn_type = 'lstm'
    #     rnn_hidden_size = 64
    #     rnn_num_layers = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        save_interval = 100
        # policy_class_name = "ActorCriticRecurrent"
        max_iterations = 15000
        run_name = ''
        experiment_name = 'go2'

  
