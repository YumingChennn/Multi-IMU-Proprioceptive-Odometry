import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import argparse
import casadi as ca

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseStamped

from keyboard_controller import KeyboardController
import pinocchio

# from gamepaded import gamepad_reader
NUM_MOTOR = 12

def quat_rotate_inverse(q, v):
    """
    Rotate a vector by the inverse of a quaternion.
    Direct translation from the PyTorch version to NumPy.
    
    Args:
        q: The quaternion in (w, x, y, z) format. Shape is (..., 4).
        v: The vector in (x, y, z) format. Shape is (..., 3).
        
    Returns:
        The rotated vector in (x, y, z) format. Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    
    # Equivalent to (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    term1 = 2.0 * np.square(q_w) - 1.0
    term1_expanded = np.expand_dims(term1, axis=-1)
    a = v * term1_expanded
    
    # Equivalent to torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    q_w_expanded = np.expand_dims(q_w, axis=-1)
    b = np.cross(q_vec, v) * q_w_expanded * 2.0
    
    # Equivalent to the torch.bmm or torch.einsum operations
    # This calculates the dot product between q_vec and v
    dot_product = np.sum(q_vec * v, axis=-1)
    dot_product_expanded = np.expand_dims(dot_product, axis=-1)
    c = q_vec * dot_product_expanded * 2.0
    
    return a - b + c

def get_gravity_orientation(quaternion):
    """
    Get the gravity vector in the robot's base frame.
    Uses the exact algorithm from your PyTorch code.
    
    Args:
        quaternion: Quaternion in (w, x, y, z) format.
        
    Returns:
        3D gravity vector in the robot's base frame.
    """
    # Ensure quaternion is a numpy array
    quaternion = np.array(quaternion)
    
    # Standard gravity vector in world frame (pointing down)
    gravity_world = np.array([0, 0, -1])
    
    # Handle both single quaternion and batched quaternions
    if quaternion.shape == (4,):
        quaternion = quaternion.reshape(1, 4)
        gravity_world = gravity_world.reshape(1, 3)
        result = quat_rotate_inverse(quaternion, gravity_world)[0]
    else:
        gravity_world = np.broadcast_to(gravity_world, quaternion.shape[:-1] + (3,))
        result = quat_rotate_inverse(quaternion, gravity_world)
    return result

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def record_step_data(record_lists, sensors, gravity_b, action, ang_vel_scale, dof_vel_scale):
    record_lists["frame_pos"].append(sensors["mocap_pos_body"] * 1.0)
    record_lists["frame_vel"].append(sensors["mocap_vel_body"] * 1.0)
    record_lists["ang_vel"].append(sensors["base_ang_vel"] * ang_vel_scale)
    record_lists["acc"].append(sensors["base_acc"] * 1.0)
    record_lists["gravity_b"].append(gravity_b)
    record_lists["joint_vel"].append(sensors["qvel"] * dof_vel_scale)
    record_lists["action"].append(action)

    for leg in ("fl", "fr", "rl", "rr"):
        record_lists[f"{leg}_ang_vel"].append(sensors[f"{leg}_ang"] * 1.0)
        record_lists[f"{leg}_acc"].append(sensors[f"{leg}_acc"] * 1.0)

def compute_foot_forces(m, d):
    forces = dict.fromkeys(["FL", "FR", "RL", "RR"], 0.0)
    c_array = np.zeros(6)

    for i in range(d.ncon):
        contact = d.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        for foot in forces:
            if g1 == foot or g2 == foot:
                mujoco.mj_contactForce(m, d, i, c_array)
                forces[foot] += np.linalg.norm(c_array[:3])
    return np.array(list(forces.values()))

def extract_sensors(d):
    # print(d.sensordata[46:49])
    return {
        "qpos": d.sensordata[:12],
        "qvel": d.sensordata[12:24],
        "qtorque": d.sensordata[24:36],

        "base_quat": d.sensordata[36:40],
        "base_ang_vel": d.sensordata[40:43],
        "base_acc": d.sensordata[43:46],

        "fl_ang": d.sensordata[59:62],
        "fl_acc": d.sensordata[62:65],
        "fr_ang": d.sensordata[69:72],
        "fr_acc": d.sensordata[72:75],
        "rl_ang": d.sensordata[79:82],
        "rl_acc": d.sensordata[82:85],
        "rr_ang": d.sensordata[89:92],
        "rr_acc": d.sensordata[92:95],

        "mocap_pos_body": d.sensordata[46:49],
        "mocap_vel_body": d.sensordata[49:52],
        "mocap_quat_body": d.sensordata[36:40],
        "mocap_pos_fr": d.sensordata[95:98],
        "mocap_quat_fr": d.sensordata[104:108],
    }

def publish_ros(pub, s, foot_forces):
    pub.publish_imu(s["base_ang_vel"], s["base_acc"], "base_link", pub.publisher_base)
    pub.publish_imu(s["fl_ang"], s["fl_acc"], "FL_foot", pub.publisher_fl)
    pub.publish_imu(s["fr_ang"], s["fr_acc"], "FR_foot", pub.publisher_fr)
    pub.publish_imu(s["rl_ang"], s["rl_acc"], "RL_foot", pub.publisher_rl)
    pub.publish_imu(s["rr_ang"], s["rr_acc"], "RR_foot", pub.publisher_rr)

    pub.publish_pose(s["mocap_pos_body"], s["mocap_quat_body"], "world", pub.publisher_pose)
    pub.publish_pose(s["mocap_pos_fr"], s["mocap_quat_fr"], "world", pub.publisher_pose_fr)

    pub.publish_joint_state(
        s["qpos"], s["qvel"], s["qtorque"], foot_forces
    )

class ImuPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher')
        self.publisher_base = self.create_publisher(Imu, 'imu/data/base_link', 10)
        self.publisher_fl = self.create_publisher(Imu, 'imu/data/fl', 10)
        self.publisher_fr = self.create_publisher(Imu, 'imu/data/fr', 10)
        self.publisher_rl = self.create_publisher(Imu, 'imu/data/rl', 10)
        self.publisher_rr = self.create_publisher(Imu, 'imu/data/rr', 10)
        self.publisher_pose = self.create_publisher(PoseStamped, 'mocap/body/pose', 10)
        self.publisher_pose_fr = self.create_publisher(PoseStamped, 'mocap/fr/pose', 10)
        self.publisher_joint_state = self.create_publisher(JointState, 'joint_states', 10)
        
    def publish_imu(self, ang_vel, lin_acc, frame_id='base_link', publisher=None):
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        # Angular velocity
        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])
        
        # Linear acceleration
        msg.linear_acceleration.x = float(lin_acc[0])
        msg.linear_acceleration.y = float(lin_acc[1])
        msg.linear_acceleration.z = float(lin_acc[2])
        
        if publisher is None:
            publisher = self.publisher_base
        publisher.publish(msg)
    
    def publish_pose(self, position, quaternion, frame_id='world', publisher=None):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        # Position
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        
        # Orientation (quaternion: w, x, y, z)
        msg.pose.orientation.w = float(quaternion[0])
        msg.pose.orientation.x = float(quaternion[1])
        msg.pose.orientation.y = float(quaternion[2])
        msg.pose.orientation.z = float(quaternion[3])
        
        if publisher is None:
            publisher = self.publisher_pose
        publisher.publish(msg)
    
    def publish_joint_state(self, qpos, qvel, qtorque, foot_forces):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Position: 12 joint positions + 4 zeros = 16 total
        position = np.concatenate([qpos, np.zeros(4)])
        msg.position = position.tolist()
        
        # Velocity: 12 joint velocities + 4 zeros = 16 total
        velocity = np.concatenate([qvel, np.zeros(4)])
        msg.velocity = velocity.tolist()
        
        # Effort (torque): 12 joint torques + 4 foot forces = 16 total
        effort = np.concatenate([qtorque, foot_forces])
        msg.effort = effort.tolist()
        
        self.publisher_joint_state.publish(msg)

import numpy as np
from scipy.spatial.transform import Rotation as R

class MIPOFilter:
    def __init__(self, config):
        # config
        self.config = config
        # MIPO parameters (對應 MATLAB 的 param)
        self.mipo_use_foot_ang_contact_model = self.config.get('mipo_use_foot_ang_contact_model', 1)  # 1 = use foot gyro model, 0 = use zero velocity model

        self.mipo_conf_init()
        
        self.gravity = np.array([0, 0, -9.81])
        
        # Load Pinocchio model once
        urdf_path = "/home/ray/Multi-IMU-Proprioceptive-Odometry/mujoco_go2_him/urdf/go2.urdf"
        self.pin_model = pinocchio.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        
        # Define joint indices for each leg (3 joints per leg)
        # Go2 robot joint order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, ...
        self.leg_joint_indices = {
            0: [0, 1, 2],    # FL: hip, thigh, calf
            1: [3, 4, 5],    # FR
            2: [6, 7, 8],    # RL
            3: [9, 10, 11]   # RR
        }
        
    def mipo_conf_init(self):
        """
        Multi-IMU Proprioceptive Odometry Filter
        
        State vector (52 dims):
        - position          (1:3)
        - velocity          (4:6)
        - euler angle       (7:9)
        - foot1 pos         (10:12)
        - foot1 vel         (13:15)
        - foot2 pos         (16:18)
        - foot2 vel         (19:21)
        - foot3 pos         (22:24)
        - foot3 vel         (25:27)
        - foot4 pos         (28:30)
        - foot4 vel         (31:33)
        - body acc bias     (34:36)
        - body gyro bias    (37:39)
        - foot1 acc bias    (40:42)
        - foot2 acc bias    (43:45)
        - foot3 acc bias    (46:48)
        - foot4 acc bias    (49:51)
        - time  tk          (52)
        """
        self.state_size = 52
        # Measurement size: MATLAB uses meas_per_leg = 11 (includes 1 reserved slot per leg)
        # Each leg: pos(3) + vel(3) + gyro_vel(3) + height(1) + reserved(1) = 11
        # Total: 4 legs × 11 + 1 yaw = 45
        # Note: Index 11 per leg is reserved/unused, so only 10 measurements are actually used per leg
        self.meas_per_leg = 11  # MATLAB: param.mipo_meas_per_leg = 11
        self.meas_size = 4 * 11 + 1  # Total: 45 (matching MATLAB)

        self.ctrl_size = 19  # Control input size: gyro(3) + acc(3) + foot_acc(12) + hk(1)

        # Process noise Q1 - matching MATLAB mipo_conf_init.m exactly
        # State: pos(3), vel(3), euler(3), foot_pos(12), foot_vel(12), ba(3), bg(3), foot_ba(12), contact(1)
        self.Q1 = np.diag([
            *[1e-4]*3,      # position (1:3)
            *[1e-2]*3,      # velocity (4:6)
            *[1e-6]*3,      # euler angles (7:9)
            *[1e-4]*2,      # foot1 pos x,y
            1e-4,           # foot1 pos z
            *[1e-4]*2,      # foot1 vel x,y
            1e-4,           # foot1 vel z
            *[1e-4]*2,      # foot2 pos x,y
            1e-4,           # foot2 pos z
            *[1e-4]*2,      # foot2 vel x,y
            1e-4,           # foot2 vel z
            *[1e-4]*2,      # foot3 pos x,y
            1e-4,           # foot3 pos z
            *[1e-4]*2,      # foot3 vel x,y
            1e-4,           # foot3 vel z
            *[1e-4]*2,      # foot4 pos x,y
            1e-4,           # foot4 pos z
            *[1e-4]*2,      # foot4 vel x,y
            1e-4,           # foot4 vel z
            *[1e-4]*3,      # body acc bias (34:36)
            *[1e-4]*3,      # body gyro bias (37:39)
            *[1e-4]*3,      # foot1 acc bias (40:42)
            *[1e-4]*3,      # foot2 acc bias (43:45)
            *[1e-4]*3,      # foot3 acc bias (46:48)
            *[1e-4]*3,      # foot4 acc bias (49:51)
            0               # time tk (52) - not used in KF
        ])

        # Control noise Q2: gyro(3) + acc(3) + foot_acc(12) + hk(1) = 19
        # Note: This will be updated with dt multiplication in update_control_noise()
        # MATLAB order: gyro(3), acc(3), foot1_acc(3), foot2_acc(3), foot3_acc(3), foot4_acc(3), hk(1)
        self.Q2 = np.diag([
            *[1e-4]*3,   # body gyro noise
            *[1e-4]*3,   # body acc noise
            *[1e-4]*3,   # foot1 acc noise
            *[1e-4]*3,   # foot2 acc noise
            *[1e-4]*3,   # foot3 acc noise
            *[1e-4]*3,   # foot4 acc noise
            0,           # hk (no noise)
        ])

        # Measurement noise
        # MATLAB: kf_conf.R = 1e-2*eye(kf_conf.meas_size);
        self.R = 1e-2 * np.eye(self.meas_size)
        # Will be properly set in _initialize_measurement_noise() based on mipo_use_md_test_flag

        """Setup CasADi symbolic functions for efficient computation"""
        # ========== Process Model ==========
        # Symbolic variables for state transition
        x_sym = ca.SX.sym('x', self.state_size)
        u_k_sym = ca.SX.sym('u_k', self.ctrl_size)  # MATLAB ctrl_size = 19
        u_k1_sym = ca.SX.sym('u_k1', self.ctrl_size)
        dt_sym = ca.SX.sym('dt')
        
        # Create symbolic state transition function using RK4 (對應 MATLAB dyn_rk4)
        x_next_sym = self._rk4_integration_casadi(x_sym, u_k_sym, u_k1_sym, dt_sym)
        
        # Create CasADi Function for state transition
        self.f_casadi = ca.Function('f', [x_sym, u_k_sym, u_k1_sym, dt_sym], [x_next_sym])
        
        # Create Jacobians using automatic differentiation
        # F = ∂f/∂x (52×52)
        self.F_jac_casadi = ca.Function('F_jac', [x_sym, u_k_sym, u_k1_sym, dt_sym], 
                                        [ca.jacobian(x_next_sym, x_sym)])
        
        # B = ∂f/∂u_k (52×19) - only w.r.t u_k, dt is a separate parameter
        self.B_jac_casadi = ca.Function('B_jac', [x_sym, u_k_sym, u_k1_sym, dt_sym],
                                        [ca.jacobian(x_next_sym, u_k_sym)])
        
        # ========== Measurement Model ==========
        # Symbolic variables for measurements (對應 MATLAB 的符號變量)
        joint_angles_sym = ca.SX.sym('phik', 12)      # s_phik
        wk_sym = ca.SX.sym('wk', 3)                    # s_wk (body angular velocity)
        joint_vels_sym = ca.SX.sym('dphik', 12)       # s_dphik
        yawk_sym = ca.SX.sym('yaw', 1)                 # s_yawk (yaw measurement)
        gyro_feet_sym = ca.SX.sym('foot_gyro', 12)    # s_foot_gyrok
        
        # Create symbolic measurement function (對應 s_r = mipo_measurement(...))
        y_sym = self._measurement_model_casadi(x_sym, wk_sym, joint_angles_sym, joint_vels_sym, yawk_sym, gyro_feet_sym)
        
        # Measurement function (對應 kf_conf.r)
        self.h_casadi = ca.Function('meas', 
                                    [x_sym, wk_sym, joint_angles_sym, joint_vels_sym, yawk_sym, gyro_feet_sym], 
                                    [y_sym])
        
        # Measurement Jacobian H = ∂h/∂x (對應 kf_conf.dr)
        H_sym = ca.jacobian(y_sym, x_sym)
        self.H_jac_casadi = ca.Function('meas_jac', 
                                        [x_sym, wk_sym, joint_angles_sym, joint_vels_sym, yawk_sym, gyro_feet_sym],
                                        [H_sym])
        
        # Foot IMU frame transformations (R_fm: IMU frame -> Foot center frame)
        # FL, FR, RL, RR
        self.R_fm_list = [
            # FL (Front Left)
            np.array([[-1,  0,  0],
                     [ 0,  0, -1],
                     [ 0, -1,  0]]),
            # FR (Front Right)
            np.array([[-1,  0,  0],
                     [ 0,  0,  1],
                     [ 0,  1,  0]]),
            # RL (Rear Left)
            np.array([[-1,  0,  0],
                     [ 0,  0, -1],
                     [ 0, -1,  0]]),
            # RR (Rear Right)
            np.array([[-1,  0,  0],
                     [ 0,  0,  1],
                     [ 0,  1,  0]])
        ] 

    def initialize_state(self, init_pos, init_euler, foot_positions):
        """Initialize filter state"""
        # State: [pos(3), vel(3), euler(3), foot_pos(12), foot_vel(12), 
        #         ba(3), bg(3), foot_ba(12), contact(1)]
        self.x = np.zeros(self.state_size)

        self.x[0:3] = init_pos     # position
        self.x[3:6] = np.zeros(3)  # velocity
        self.x[6:9] = init_euler   # orientation (euler)
        self.x[9:21] = foot_positions.flatten()  # 4 feet * 3
        self.x[21:33] = np.zeros(12)  # foot velocities
        self.x[33:36] = np.zeros(3)  # biases
        self.x[36:39] = np.zeros(3)  # biases
        self.x[39:42] = np.zeros(3)  # biases
        self.x[42:45] = np.zeros(3)  # biases
        self.x[45:48] = np.zeros(3)  # biases
        self.x[48:51] = np.zeros(3)  # biases
        self.x[51:52] = np.zeros(1)

        self.P = np.eye(self.state_size) * self.config.get('init_cov', 0.1)
        self.P[34:52, 34:52] = np.eye(18) * self.config.get('init_bias_cov', 1e-4)

    def transform_foot_imu_to_body(self, joint_angles, gyro_feet_raw, acc_feet_raw):
        """
        Transform foot IMU measurements from IMU frame to body frame
        Following MATLAB implementation:
        accel_IMU_bs = R_bf * R_fm * accel_IMUs
        gyro_IMU_bs = R_bf * R_fm * gyro_IMUs
        
        Args:
            joint_angles: All joint angles (12,) = [FL(3), FR(3), RL(3), RR(3)]
            gyro_feet_raw: Raw gyro measurements in IMU frame (12,)
            acc_feet_raw: Raw accelerometer measurements in IMU frame (12,)
            
        Returns:
            gyro_feet_body: Gyro in body frame (12,)
            acc_feet_body: Acceleration in body frame (12,)
        """
        gyro_feet_body = np.zeros(12)
        acc_feet_body = np.zeros(12)
        
        for leg_id in range(4):
            # Get joint angles for this leg
            leg_joint_angs = joint_angles[leg_id*3:(leg_id+1)*3]
            
            # Get rotation from body to foot center frame (R_bf)
            R_bf = self.get_foot_rotation(leg_joint_angs, leg_id)
            
            # Get fixed rotation from foot center to IMU frame (R_fm)
            R_fm = self.R_fm_list[leg_id]
            
            # Get raw IMU measurements for this leg
            gyro_imu = gyro_feet_raw[leg_id*3:(leg_id+1)*3]
            acc_imu = acc_feet_raw[leg_id*3:(leg_id+1)*3]
            
            # Transform: IMU frame -> Foot center frame -> Body frame
            # accel_IMU_bs = R_bf * R_fm * accel_IMUs
            # gyro_IMU_bs = R_bf * R_fm * gyro_IMUs
            gyro_feet_body[leg_id*3:(leg_id+1)*3] = R_bf @ R_fm @ gyro_imu
            acc_feet_body[leg_id*3:(leg_id+1)*3] = R_bf @ R_fm @ acc_imu
        
        return gyro_feet_body, acc_feet_body
    
    def get_foot_rotation(self, angles, leg_id):
        """
        計算足部中心相對於機體的旋轉矩陣
        對應 MATLAB 的 autoFunc_fk_pf_rot
        
        Args:
            angles: Joint angles [hip, thigh, calf] (3,)
            leg_id: Leg identifier 0=FL, 1=FR, 2=RL, 3=RR
            
        Returns:
            R_bf: Rotation matrix from body frame to foot center frame (3x3)
            
        MATLAB equivalent:
            R_bf = autoFunc_fk_pf_rot(angles, lc, rho_fix)
        """
        # Extract joint angles
        t1, t2, t3 = angles[0], angles[1], angles[2]
        
        # Precompute trig functions (matching MATLAB)
        t5 = np.cos(t1)
        t6 = np.sin(t1)
        t7 = t2 + t3
        t8 = np.cos(t7)
        t9 = np.sin(t7)
        
        # MATLAB formula: R_bf = reshape([t8,t6.*t9,-t5.*t9,0.0,t5,t6,t9,-t6.*t8,t5.*t8],[3,3])
        # reshape fills column by column in MATLAB
        R_bf = np.array([
            [t8,      0.0,  t9     ],
            [t6*t9,   t5,   -t6*t8 ],
            [-t5*t9,  t6,   t5*t8  ]
        ])
        
        return R_bf
    
    def predict(self, u_k, u_k1, dt):
        """EKF Prediction step"""
        # State prediction
        x_pred = self.state_transition(self.x, u_k, u_k1, dt)
        
        # Jacobian F (state transition)
        F = self.compute_F_jacobian(self.x, u_k, u_k1, dt)
        
        # Jacobian B (control noise)
        B = self.compute_B_jacobian(self.x, u_k, u_k1, dt)
        
        # Update process noise based on dt
        self.update_process_noise(dt)
        
        # Update control noise based on dt (matching MATLAB)
        self.update_control_noise(dt)

        self._initialize_measurement_noise(dt)
        
        # Covariance prediction
        P_pred = F @ self.P @ F.T + self.Q1 + B @ self.Q2 @ B.T
        # P_pred = (P_pred + P_pred.T) / 2  # Ensure symmetry
        
        return x_pred, P_pred
    
    def state_transition(self, x, u_k, u_k1, dt):
        """
        State transition function f(x, u) - using CasADi for efficiency
        x: current state
        u_k: [gyro_body(3), acc_body(3), acc_feet(12), hk(1)] (19 dims)
        u_k1: next timestep control
        """
        # Use CasADi function for state transition
        x_new = self.f_casadi(x, u_k, u_k1, dt).full().flatten()
        return x_new
    
    def compute_F_jacobian(self, x, u_k, u_k1, dt):
        """Compute state transition Jacobian using CasADi automatic differentiation"""
        # Use CasADi automatic differentiation
        F = self.F_jac_casadi(x, u_k, u_k1, dt).full()
        return F
    
    def compute_B_jacobian(self, x, u_k, u_k1, dt):
        """
        Compute control Jacobian using CasADi automatic differentiation
        
        B = ∂f/∂u where u = [gyro(3), acc(3), foot_acc(12), hk(1)]
        
        Args:
            x: Current state (52,)
            u_k: Current control input (19,) - [gyro(3), acc(3), foot_acc(12), hk(1)]
            u_k1: Next control input (19,)
            dt: Time step (separate parameter, not in u_k)
            
        Returns:
            B: Control Jacobian (52, 19)
        """
        # Use CasADi automatic differentiation
        B = self.B_jac_casadi(x, u_k, u_k1, dt).full()
        return B
    
    def update_process_noise(self, dt):
        """Update Q1 based on dt - matching MATLAB exactly"""
        cfg = self.config
        
        # Build Q1 diagonal matching MATLAB structure
        q1_diag = []
        
        # Position (3): x, y use proc_n_pos; z uses proc_n_pos
        q1_diag.extend([cfg['proc_n_pos'] * dt] * 2)   # pos x, y
        q1_diag.append(cfg['proc_n_pos'] * dt)         # pos z
        
        # Velocity (3): x, y use proc_n_vel_xy; z uses proc_n_vel_z
        q1_diag.append(cfg['proc_n_vel_xy'] * dt)      # vel x
        q1_diag.append(cfg['proc_n_vel_xy'] * dt)      # vel y
        q1_diag.append(cfg['proc_n_vel_z'] * dt)       # vel z
        
        # Euler angles (3)
        q1_diag.extend([cfg['proc_n_ang'] * dt] * 3)
        
        # Foot positions and velocities (24 total = 4 legs × 6)
        # MATLAB: repmat([pos_xy(2), pos_z(1), vel_xy(2), vel_z(1)], 4, 1)
        for leg in range(4):
            q1_diag.extend([cfg['proc_n_foot_pos'] * dt] * 2)  # foot pos x, y
            q1_diag.append(cfg['proc_n_foot_pos'] * dt)        # foot pos z
            q1_diag.extend([cfg['proc_n_foot_vel'] * dt] * 2)  # foot vel x, y
            q1_diag.append(cfg['proc_n_foot_vel'] * dt)        # foot vel z
        
        # Body biases (6)
        q1_diag.extend([cfg['proc_n_ba'] * dt] * 3)    # body acc bias
        q1_diag.extend([cfg['proc_n_bg'] * dt] * 3)    # body gyro bias
        
        # Foot acc biases (12 = 4 legs × 3)
        q1_diag.extend([cfg['proc_n_foot1_ba'] * dt]*3)  # foot1 acc bias
        q1_diag.extend([cfg['proc_n_foot2_ba'] * dt]*3)  # foot2 acc bias
        q1_diag.extend([cfg['proc_n_foot3_ba'] * dt]*3)  # foot3 acc bias   
        q1_diag.extend([cfg['proc_n_foot4_ba'] * dt]*3)  # foot4 acc bias
        
        # Time (1)
        q1_diag.append(0)
        
        self.Q1 = np.diag(q1_diag)
    
    def update_control_noise(self, dt):
        """Update Q2 based on dt - matching MATLAB exactly"""
        cfg = self.config
        
        # Build Q2 diagonal matching MATLAB structure
        # MATLAB order: acc(3), gyro(3), foot1_acc(3), foot2_acc(3), foot3_acc(3), foot4_acc(3), hk(1)
        # Note: MATLAB uses [ctrl_n_acc, ctrl_n_gyro, ctrl_n_foot1_acc, ...]
        q2_diag = []
        
        # Body control noise
        q2_diag.extend([cfg['ctrl_n_gyro'] * dt] * 3)  # body gyro
        q2_diag.extend([cfg['ctrl_n_acc'] * dt] * 3)   # body acc
        
        # Foot accelerometer noise
        q2_diag.extend([cfg['ctrl_n_foot1_acc'] * dt] * 3)  # foot1 acc
        q2_diag.extend([cfg['ctrl_n_foot2_acc'] * dt] * 3)  # foot2 acc
        q2_diag.extend([cfg['ctrl_n_foot3_acc'] * dt] * 3)  # foot3 acc
        q2_diag.extend([cfg['ctrl_n_foot4_acc'] * dt] * 3)  # foot4 acc
        
        # hk (no noise)
        q2_diag.append(0)
        
        self.Q2 = np.diag(q2_diag)
    
    def _initialize_measurement_noise(self, dt):
        """Initialize R matrix matching MATLAB when mipo_use_md_test_flag=1
        
        MATLAB code:
            for i = 1:param.num_leg
                mipo_conf.R((i-1)*num_meas+7:(i-1)*num_meas+9,(i-1)*num_meas+7:(i-1)*num_meas+9) = 
                    param.meas_n_zero_vel *eye(3);  % 0.01
                mipo_conf.R((i-1)*num_meas+10,(i-1)*num_meas+10) = 
                    param.meas_n_foot_height;  % 0.001
            end
        """
        cfg = self.config
        
        # Keep R as 1e-2 * I for non-critical measurements (matching MATLAB initial value)
        # Only update the critical measurements below
        
        # Set measurement noise for each leg
        for i in range(4):  # 4 legs
            # Zero velocity constraint: indices 6:9 per leg (Python 0-indexed)
            # MATLAB: (i-1)*11+7:(i-1)*11+9
            idx_vel_start = i * self.meas_per_leg + 6
            self.R[idx_vel_start:idx_vel_start+3, idx_vel_start:idx_vel_start+3] = \
                cfg['meas_n_zero_vel'] * np.eye(3)  # 0.01 * I
            
            # Foot height constraint: index 9 per leg (Python 0-indexed)
            # MATLAB: (i-1)*11+10
            idx_height = i * self.meas_per_leg + 9
            self.R[idx_height, idx_height] = cfg['meas_n_foot_height']  # 0.001
      
    def update(self, x_pred, P_pred, wk, joint_angles, joint_vels, yawk, gyro_feet, contact_flags):
        """EKF Update step
        
        Args:
            x_pred: Predicted state
            P_pred: Predicted covariance
            wk: Body angular velocity (3,)
            joint_angles: Joint angles (12,)
            joint_vels: Joint velocities (12,)
            yawk: Yaw measurement (scalar)
            gyro_feet: Foot gyro measurements (12,)
            contact_flags: Contact flags (4,)
        """
        
        # Predicted measurement
        y_pred = self.measurement_model(x_pred, wk, joint_angles, joint_vels, yawk, gyro_feet)
        
        # Measurement Jacobian
        H = self.compute_H_jacobian(x_pred, wk, joint_angles, joint_vels, yawk, gyro_feet)
        
        # Innovation covariance
        S = H @ P_pred @ H.T + self.R
        
        # Measurement residual y (MATLAB已經計算好residual)
        # MATLAB: y = full(mipo_conf.r(x01, hat_wk, hat_phik, hat_dphik, hat_yawk, gyro_IMU_bs))
        # Note: measurement_model已經返回殘差，不是預測值
        y = y_pred  # 實際上是residual
        
        # Mahalanobis distance test (outlier rejection)
        # MATLAB: mask = ones(meas_size,1); then set mask=0 if MD > 4
        mask = self.mahalanobis_test(y, S, contact_flags)
        
        # Kalman gain computation
        # MATLAB: update = P01 * H(mask,:)' * (S(mask,mask)\y(mask))
        H_masked = H[mask, :]
        S_masked = S[np.ix_(mask, mask)]
        y_masked = y[mask]
        
        K_masked = P_pred @ H_masked.T @ np.linalg.inv(S_masked)
        update = K_masked @ y_masked
        
        # State update
        # MATLAB: x_list(:,k+1) = x01 - update
        # 注意：用減號，因為y已經是residual (prediction - actual)
        self.x = x_pred - update
        
        # Covariance update - matching MATLAB exactly
        # MATLAB: cov_list(:,:,k+1) = (eye(state_size) - P01*H(mask,:)'*(S(mask,mask)\H(mask,:))) * P01
        self.P = (np.eye(self.state_size) - P_pred @ H_masked.T @ np.linalg.inv(S_masked) @ H_masked) @ P_pred
        
        # Ensure symmetry
        # MATLAB: cov_list(:,:,k+1) = (cov_list(:,:,k+1) + cov_list(:,:,k+1)')/2
        self.P = (self.P + self.P.T) / 2

    def measurement_model(self, x, wk, joint_angles, joint_vels, yawk, gyro_feet):
        """
        Measurement model h(x) - NumPy version
        對應 MATLAB 的 mipo_measurement
        
        Args:
            x: State (52,)
            wk: Body angular velocity (3,)
            joint_angles: Joint angles (12,)
            joint_vels: Joint velocities (12,)
            yawk: Yaw measurement (scalar)
            gyro_feet: Foot gyro measurements (12,)
            
        Returns: Measurement residuals (41,)
        """
        # Note: This NumPy version is for reference/debugging
        # The actual filter uses the CasADi version via h_casadi
        # For now, just call the CasADi version and convert to NumPy
        y_casadi = self.h_casadi(x, wk, joint_angles, joint_vels, yawk, gyro_feet)
        return np.array(y_casadi.full()).flatten()

    def compute_H_jacobian(self, x, wk, joint_angles, joint_vels, yawk, gyro_feet):
        """
        Compute measurement Jacobian using CasADi automatic differentiation
        對應 MATLAB 的 kf_conf.dr
        
        Args:
            x: State (52,)
            wk: Body angular velocity (3,)
            joint_angles: Joint angles (12,)
            joint_vels: Joint velocities (12,)
            yawk: Yaw measurement (1,)
            gyro_feet: Foot gyro measurements (12,)
        """
        # 使用 CasADi 自動微分（快速且精確）
        H = self.H_jac_casadi(x, wk, joint_angles, joint_vels, yawk, gyro_feet).full()
        return H
    
    def mahalanobis_test(self, innovation, S, contact_flags, threshold=3.0):
        """Mahalanobis distance test for outlier rejection - matching MATLAB (threshold=3)
        
        MATLAB code:
            if MD > 3
                mask((i-1)*num_meas+7:(i-1)*num_meas+9) = zeros(3,1);
            end
        """
        mask = np.ones(len(innovation), dtype=bool)
        
        # MATLAB: for i = 1:param.num_leg
        #   seg_mes = y((i-1)*num_meas+7:(i-1)*num_meas+9);
        for i in range(4):
            # Test zero velocity constraint (indices 6:9 per leg in Python)
            # MATLAB uses (i-1)*11+7:(i-1)*11+9
            idx_start = i * self.meas_per_leg + 6  # Python 0-indexed
            idx_end = idx_start + 3
            seg_innov = innovation[idx_start:idx_end]
            seg_S = S[idx_start:idx_end, idx_start:idx_end]
            
            MD = np.sqrt(seg_innov.T @ np.linalg.inv(seg_S) @ seg_innov)
            if MD > threshold:
                mask[idx_start:idx_end] = np.zeros(3, dtype=bool)  # Set to False for outlier rejection
                
        return mask
    
    def _rk4_integration_casadi(self, x, u_k, u_k1, dt):
        """
        RK4 (Runge-Kutta 4th order) integration for state transition
        對應 MATLAB 的 dyn_rk4 函數
        
        Args:
            x: State at time k (xn)
            u_k: Control at time k (un)
            u_k1: Control at time k+1 (un1)
            dt: Time step
            
        Returns:
            x_next: State at time k+1 (xn1)
            
        MATLAB implementation:
            k1 = dynfunc(xn, un);
            k2 = dynfunc(xn+dt*k1/2, (un+un1)/2);
            k3 = dynfunc(xn+dt*k2/2, (un+un1)/2);
            k4 = dynfunc(xn+dt*k3, un1);
            xn1 = xn + 1/6*dt*(k1 + 2*k2 + 2*k3 + k4);
        """
        # Average control for intermediate steps
        u_avg = 0.5 * (u_k + u_k1)
        
        # k1 = f(x_k, u_k)
        k1 = self._process_dynamics_casadi(x, u_k)
        
        # k2 = f(x_k + dt*k1/2, (u_k+u_k1)/2)
        k2 = self._process_dynamics_casadi(x + dt*k1/2, u_avg)
        
        # k3 = f(x_k + dt*k2/2, (u_k+u_k1)/2)
        k3 = self._process_dynamics_casadi(x + dt*k2/2, u_avg)
        
        # k4 = f(x_k + dt*k3, u_k1)
        k4 = self._process_dynamics_casadi(x + dt*k3, u_k1)
        
        # Final RK4 update: xn1 = xn + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x_next
    
    def _process_dynamics_casadi(self, x, u):
        """
        Process dynamics dx/dt = f(x, u)
        對應 MATLAB 的 mipo_process_dyn 函數
        
        Args:
            x: State (52,)
            u: Control input (19,) - [gyro(3), acc(3), foot_acc(12), hk(1)]
            
        Returns:
            x_dot: State derivative (52,)
        """
        # Extract state - matching MATLAB's explicit indexing
        pos = x[0:3]
        vel = x[3:6]
        euler = x[6:9]

        # MATLAB state order: foot1_pos, foot1_vel, foot2_pos, foot2_vel, ...
        foot1_pos = x[9:12]
        foot1_vel = x[12:15]
        foot2_pos = x[15:18]
        foot2_vel = x[18:21]
        foot3_pos = x[21:24]
        foot3_vel = x[24:27]
        foot4_pos = x[27:30]
        foot4_vel = x[30:33]

        ba = x[33:36]
        bg = x[36:39]
        
        foot1_ba = x[39:42]
        foot2_ba = x[42:45]
        foot3_ba = x[45:48]
        foot4_ba = x[48:51]
        
        # Extract control (with bias correction) - matching MATLAB
        gyro_body = u[0:3] - bg
        acc_body = u[3:6] - ba
        
        foot1_acc = u[6:9] - foot1_ba
        foot2_acc = u[9:12] - foot2_ba
        foot3_acc = u[12:15] - foot3_ba
        foot4_acc = u[15:18] - foot4_ba

        # Orientation derivative
        euler_dot = self._euler_rate_casadi(euler, gyro_body)
        
        # Rotation matrix from body frame to world frame
        R_wb = self._euler_to_rot_casadi(euler)
        
        # Body dynamics
        # MATLAB: acc = R*a - [0;0;9.8]
        # Note: gravity points down, so we subtract positive gravity
        gravity_vec = ca.vertcat(0, 0, 9.81)
        acc_world = ca.mtimes(R_wb, acc_body) - gravity_vec
        
        # Foot dynamics - matching MATLAB
        foot1_pos_dot = foot1_vel
        foot2_pos_dot = foot2_vel
        foot3_pos_dot = foot3_vel
        foot4_pos_dot = foot4_vel
        
        # MATLAB: foot_acc_w = R*foot_acc - [0;0;9.8]
        foot1_vel_dot = ca.mtimes(R_wb, foot1_acc) - gravity_vec
        foot2_vel_dot = ca.mtimes(R_wb, foot2_acc) - gravity_vec
        foot3_vel_dot = ca.mtimes(R_wb, foot3_acc) - gravity_vec
        foot4_vel_dot = ca.mtimes(R_wb, foot4_acc) - gravity_vec
        
        # Concatenate all derivatives - matching MATLAB xdot order
        # MATLAB: [vel; acc; deuler;
        #          foot1_vel; foot1_acc_w;
        #          foot2_vel; foot2_acc_w;
        #          foot3_vel; foot3_acc_w;
        #          foot4_vel; foot4_acc_w;
        #          zeros(3,1); zeros(3,1); zeros(3,1); zeros(3,1); zeros(3,1); zeros(3,1); 1]
        x_dot = ca.vertcat(
            vel,              # vel (derivative of pos)
            acc_world,              # acc (derivative of vel)
            euler_dot,            # deuler
            foot1_pos_dot,        # foot1_vel (derivative of foot1_pos)
            foot1_vel_dot,        # foot1_acc_w (derivative of foot1_vel)
            foot2_pos_dot,        # foot2_vel
            foot2_vel_dot,        # foot2_acc_w
            foot3_pos_dot,        # foot3_vel
            foot3_vel_dot,        # foot3_acc_w
            foot4_pos_dot,        # foot4_vel
            foot4_vel_dot,        # foot4_acc_w
            ca.SX.zeros(3, 1),               # zeros(3,1) - body acc bias
            ca.SX.zeros(3, 1),               # zeros(3,1) - body gyro bias
            ca.SX.zeros(12, 1),          # zeros(12,1) - foot acc biases
            ca.SX.ones(1, 1)           # 1
        )
        
        return x_dot
    
    @staticmethod
    def _euler_rate_casadi(euler, omega):
        """
        Convert body angular velocity to Euler angle rates
        euler_dot = T(euler) * omega
        
        Args:
            euler: Euler angles [roll, pitch, yaw]
            omega: Body angular velocity [wx, wy, wz]
            
        Returns:
            euler_dot: Euler angle rates
        """
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        # Transformation matrix from body rates to Euler rates
        cr = ca.cos(roll)
        sr = ca.sin(roll)
        cp = ca.cos(pitch)
        tp = ca.tan(pitch)
        
        T = ca.vertcat(
            ca.horzcat(1, sr*tp, cr*tp),
            ca.horzcat(0, cr, -sr),
            ca.horzcat(0, sr/cp, cr/cp)
        )
        
        euler_dot = ca.mtimes(T, omega)
        return euler_dot
    
    @staticmethod
    def _euler_to_rot_casadi(euler):
        """Convert euler angles to rotation matrix using CasADi"""
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        # Roll (rotation around X-axis)
        cr = ca.cos(roll)
        sr = ca.sin(roll)
        R_x = ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, cr, -sr),
            ca.horzcat(0, sr, cr)
        )
        
        # Pitch (rotation around Y-axis)
        cp = ca.cos(pitch)
        sp = ca.sin(pitch)
        R_y = ca.vertcat(
            ca.horzcat(cp, 0, sp),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-sp, 0, cp)
        )
        
        # Yaw (rotation around Z-axis)
        cy = ca.cos(yaw)
        sy = ca.sin(yaw)
        R_z = ca.vertcat(
            ca.horzcat(cy, -sy, 0),
            ca.horzcat(sy, cy, 0),
            ca.horzcat(0, 0, 1)
        )
        
        # Combined rotation: R = Rz * Ry * Rx
        return ca.mtimes([R_z, R_y, R_x])
   
    @staticmethod
    def euler_to_rot(euler):
        """
        Convert euler angles to rotation matrix (NumPy version)
        Matches CasADi version: R = Rz * Ry * Rx
        
        Args:
            euler: [roll, pitch, yaw] (3,)
            
        Returns:
            R: 3×3 rotation matrix (NumPy)
        """
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        
        # Roll (rotation around X-axis)
        cr = np.cos(roll)
        sr = np.sin(roll)
        R_x = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])
        
        # Pitch (rotation around Y-axis)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        R_y = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ])
        
        # Yaw (rotation around Z-axis)
        cy = np.cos(yaw)
        sy = np.sin(yaw)
        R_z = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = Rz * Ry * Rx (matching CasADi version)
        return R_z @ R_y @ R_x
    
    def _measurement_model_casadi(self, x, wk, joint_angles, joint_vels, yawk, gyro_feet):
        """
        Measurement model using CasADi symbolic expressions
        對應 MATLAB 的 mipo_measurement 函數
        
        Args:
            x: State (52,)
            wk: Body angular velocity (3,) - from IMU
            joint_angles: Joint angles (12,) - phik
            joint_vels: Joint velocities (12,) - dphik
            yawk: Yaw measurement (1,) - from IMU or magnetometer
            gyro_feet: Foot gyroscope measurements (12,) - foot_gyro
        
        Returns:
            y: Measurement residuals (45,) = 4 legs × 11 + 1 yaw
            Per leg (11): pos_residual(3) + vel_residual(3) + gyro_vel_residual(3) + height_residual(1) + unused(1)
        """
        # Extract state - matching MATLAB's explicit indexing
        pos = x[0:3]
        vel = x[3:6]
        euler = x[6:9]

        # Rotation matrix
        R_wb = self._euler_to_rot_casadi(euler)  # R_er in MATLAB
        R_bw = R_wb.T  # R_er' in MATLAB (body to world)
        
        foot1_pos = x[9:12]
        foot1_vel = x[12:15]
        foot2_pos = x[15:18]
        foot2_vel = x[18:21]
        foot3_pos = x[21:24]
        foot3_vel = x[24:27]
        foot4_pos = x[27:30]
        foot4_vel = x[30:33]

        # Create list of foot positions and velocities for easier access
        foot_positions = [foot1_pos, foot2_pos, foot3_pos, foot4_pos]
        foot_velocities = [foot1_vel, foot2_vel, foot3_vel, foot4_vel]
        bg = x[36:39]

        # Initialize measurement residual list
        residual_list = []
        
        # MATLAB: meas_per_leg = 11, but only uses indices 1-10
        for i in range(4):
            # Extract angles and velocities for this leg
            angles = joint_angles[i*3:(i+1)*3]
            ang_vels = joint_vels[i*3:(i+1)*3]
            
            # Forward kinematics in body frame (p_rf in MATLAB)
            p_rf = self._forward_kinematics_symbolic(angles, i)
            
            # Jacobian for velocity calculation
            # Use exact MATLAB formula instead of automatic differentiation
            J_rf = self._forward_kinematics_jacobian(angles, i)
            
            # Leg velocity in body frame considering rotation
            # MATLAB: leg_v = J_rf*av + skew(wk-bg)*p_rf
            omega_corrected = wk - bg
            skew_omega = self._skew_symmetric(omega_corrected)
            leg_v = ca.mtimes(J_rf, ang_vels) + ca.mtimes(skew_omega, p_rf)
            
            # Get foot state from state vector
            foot_pos_i = foot_positions[i]
            foot_vel_i = foot_velocities[i]
            
            # Measurement residual 1-3: FK position residual
            # MATLAB: meas_residual((i-1)*11+1:(i-1)*11+3) = p_rf - R_er'*(foot_pos - pos)
            pos_residual = p_rf - ca.mtimes(R_bw, (foot_pos_i - pos))
            residual_list.append(pos_residual)
            
            # Measurement residual 4-6: FK velocity residual
            # MATLAB: meas_residual((i-1)*11+4:(i-1)*11+6) = foot_vel - (vel + R_er*leg_v)
            vel_residual = foot_vel_i - (vel + ca.mtimes(R_wb, leg_v))
            residual_list.append(vel_residual)
            
            # Measurement residual 7-9: Foot gyro-based velocity constraint
            # MATLAB: Uses foot gyro to calculate velocity
            foot_w_robot = gyro_feet[i*3:(i+1)*3]
            foot_w_world = ca.mtimes(R_wb, foot_w_robot)
            p_rf_world = ca.mtimes(R_wb, p_rf)
            
            # Foot support vector (假設足端有 5cm 的支撐長度)
            p_rf_norm = ca.norm_2(p_rf_world)
            foot_support_vec = -p_rf_world / p_rf_norm * 0.05
            foot_vel_from_gyro = ca.cross(foot_w_world, foot_support_vec)
            
            # MATLAB: Conditional based on mipo_use_foot_ang_contact_model
            if self.mipo_use_foot_ang_contact_model == 1:
                # Use foot gyro to constrain velocity
                # MATLAB: meas_residual((i-1)*11+7:(i-1)*11+9) = foot_vel - foot_vel_world
                gyro_vel_residual = foot_vel_i - foot_vel_from_gyro
            else:
                # Use zero velocity constraint (simplified)
                # MATLAB: meas_residual((i-1)*11+7:(i-1)*11+9) = foot_vel
                gyro_vel_residual = foot_vel_i
            
            residual_list.append(gyro_vel_residual)
            
            # Measurement residual 10: Foot height constraint
            # MATLAB: meas_residual((i-1)*11+10) = foot_pos((i-1)*3+3) 
            height_residual = foot_pos_i[2]
            residual_list.append(height_residual)
            
            # Measurement residual 11: Reserved/unused slot
            # MATLAB defines meas_per_leg=11 but doesn't use index 11
            # Keep dimension consistent with MATLAB - this slot is never updated in MATLAB
            reserved_residual = ca.SX.zeros(1, 1)  # Scalar zero
            residual_list.append(reserved_residual)
        
        # Yaw measurement residual
        # MATLAB: meas_residual(end) = yawk - euler(3)
        yaw_residual = yawk - euler[2]
        residual_list.append(yaw_residual)
        
        # Concatenate all measurement residuals
        # Total: 4 legs × 11 residuals + 1 yaw = 45 dimensions (matching MATLAB)
        return ca.vertcat(*residual_list)
    
    def _forward_kinematics_symbolic(self, angles, leg_id):
        """
        符號版本的正向運動學 - 對應 MATLAB 的 autoFunc_fk_pf_pos
        使用 Go2 機器人的實際運動學參數
        
        Args:
            angles: Joint angles [hip, thigh, calf] (3,) - CasADi symbolic
            leg_id: Leg identifier 0=FL, 1=FR, 2=RL, 3=RR
            
        Returns:
            foot_pos: Foot position in body frame (3,) - CasADi symbolic
            
        MATLAB equivalent:
            p_bf = autoFunc_fk_pf_pos(angles, lc, rho_fix)
            where rho_fix = [ox; oy; d; lt]
        """

        # Go2 robot parameters from URDF (對應 MATLAB 的 param)
        # FL, FR, RL, RR
        ox_list = [0.1934, 0.1934, -0.1934, -0.1934]   # Hip offset X
        oy_list = [0.0465, -0.0465, 0.0465, -0.0465]   # Hip offset Y
        d_list = [0.0955, -0.0955, 0.0955, -0.0955]     # Hip offset Z (abad link length)
        
        lt = 0.213  # Thigh length
        lc = 0.213  # Calf length
        
        # Get parameters for this leg (對應 MATLAB 的 rho_fix(:,i))
        ox = ox_list[leg_id]
        oy = oy_list[leg_id]
        d = d_list[leg_id]
        
        # MATLAB formula (符號數學工具箱生成的精確公式):
        # p_bf = [ox-lt.*t9-lc.*sin(t2+t3);
        #         oy+d.*t5+lt.*t6.*t8+lc.*t6.*t7.*t8-lc.*t8.*t9.*t10;
        #         d.*t8-lt.*t5.*t6-lc.*t5.*t6.*t7+lc.*t5.*t9.*t10];
        # where:
        #   t5 = cos(t1), t6 = cos(t2), t7 = cos(t3)
        #   t8 = sin(t1), t9 = sin(t2), t10 = sin(t3)
        
        # Extract joint angles (對應 MATLAB 的 t1, t2, t3)
        t1 = angles[0]  # hip angle
        t2 = angles[1]  # thigh angle
        t3 = angles[2]  # calf angle
        t5 = ca.cos(t1)
        t6 = ca.cos(t2)
        t7 = ca.cos(t3)
        t8 = ca.sin(t1)
        t9 = ca.sin(t2)
        t10 = ca.sin(t3)
        
        # Exact MATLAB formula
        x_bf = ox - lt*t9 - lc*ca.sin(t2 + t3)
        y_bf = oy + d*t5 + lt*t6*t8 + lc*t6*t7*t8 - lc*t8*t9*t10
        z_bf = d*t8 - lt*t5*t6 - lc*t5*t6*t7 + lc*t5*t9*t10
        
        return ca.vertcat(x_bf, y_bf, z_bf)
    
    def _forward_kinematics_jacobian(self, angles, leg_id):
        """
        Jacobian of forward kinematics - 對應 MATLAB 的 autoFunc_d_fk_dt
        使用 Go2 機器人的實際運動學參數
        
        Args:
            angles: Joint angles [hip, thigh, calf] (3,) - CasADi symbolic
            leg_id: Leg identifier 0=FL, 1=FR, 2=RL, 3=RR
            
        Returns:
            J: Jacobian matrix (3x3) - ∂p_bf/∂angles
            
        MATLAB equivalent:
            J = autoFunc_d_fk_dt(angles, lc, rho_fix)
            where rho_fix = [ox; oy; d; lt]
        """
        # Go2 robot parameters from URDF
        ox_list = [0.1934, 0.1934, -0.1934, -0.1934]
        oy_list = [0.0465, -0.0465, 0.0465, -0.0465]
        d_list = [0.0955, -0.0955, 0.0955, -0.0955]
        
        lt = 0.213  # Thigh length
        lc = 0.213  # Calf length
        
        # Get parameters for this leg
        ox = ox_list[leg_id]
        oy = oy_list[leg_id]
        d = d_list[leg_id]
        
        # Extract joint angles
        t1 = angles[0]  # hip angle
        t2 = angles[1]  # thigh angle
        t3 = angles[2]  # calf angle
        
        # Precompute trig functions (matching MATLAB variable names)
        t5 = ca.cos(t1)
        t6 = ca.cos(t2)
        t7 = ca.cos(t3)
        t8 = ca.sin(t1)
        t9 = ca.sin(t2)
        t10 = ca.sin(t3)
        t11 = t2 + t3
        t12 = ca.cos(t11)
        t13 = lt * t9
        t14 = ca.sin(t11)
        t15 = lc * t12
        t16 = lc * t14
        t17 = -t15
        t18 = t13 + t16
        
        # MATLAB formula: jacobian = reshape([...], [3,3])
        # Column 1: ∂p_bf/∂t1
        j11 = 0.0
        j21 = -d*t8 + lt*t5*t6 + lc*t5*t6*t7 - lc*t5*t9*t10
        j31 = d*t5 + lt*t6*t8 + lc*t6*t7*t8 - lc*t8*t9*t10
        
        # Column 2: ∂p_bf/∂t2
        j12 = t17 - lt*t6
        j22 = -t8*t18
        j32 = t5*t18
        
        # Column 3: ∂p_bf/∂t3
        j13 = t17
        j23 = -t8*t16
        j33 = t5*t16
        
        # Construct Jacobian matrix (3x3)
        J = ca.vertcat(
            ca.horzcat(j11, j12, j13),
            ca.horzcat(j21, j22, j23),
            ca.horzcat(j31, j32, j33)
        )
        
        return J
    
    @staticmethod
    def _skew_symmetric(v):
        """
        Create skew-symmetric matrix from vector
        Used for cross product: skew(v) * w = v × w
        
        Args:
            v: 3D vector
            
        Returns:
            3×3 skew-symmetric matrix
        """
        return ca.vertcat(
            ca.horzcat(0, -v[2], v[1]),
            ca.horzcat(v[2], 0, -v[0]),
            ca.horzcat(-v[1], v[0], 0)
        )
    
    def forward_kinematics(self, angles, leg_id):
        """
        Compute foot position from joint angles using exact MATLAB formula
        Replaces Pinocchio with analytical forward kinematics
        
        Args:
            angles: Joint angles for one leg [hip, thigh, calf] (3,)
            leg_id: Leg identifier 0=FL, 1=FR, 2=RL, 3=RR
            
        Returns:
            foot_pos: Foot position in body frame (3,) - NumPy array
        """
        # Go2 robot parameters (matching MATLAB)
        ox_list = [0.1934, 0.1934, -0.1934, -0.1934]
        oy_list = [0.0465, -0.0465, 0.0465, -0.0465]
        d_list = [0.0955, -0.0955, 0.0955, -0.0955]
        lt = 0.213
        lc = 0.213
        
        ox = ox_list[leg_id]
        oy = oy_list[leg_id]
        d = d_list[leg_id]
        
        # Extract joint angles
        t1, t2, t3 = angles[0], angles[1], angles[2]
        t5 = np.cos(t1)
        t6 = np.cos(t2)
        t7 = np.cos(t3)
        t8 = np.sin(t1)
        t9 = np.sin(t2)
        t10 = np.sin(t3)
        
        # MATLAB formula from autoFunc_fk_pf_pos
        x_bf = ox - lt*t9 - lc*np.sin(t2 + t3)
        y_bf = oy + d*t5 + lt*t6*t8 + lc*t6*t7*t8 - lc*t8*t9*t10
        z_bf = d*t8 - lt*t5*t6 - lc*t5*t6*t7 + lc*t5*t9*t10
        
        return np.array([x_bf, y_bf, z_bf])
    

if __name__ == "__main__":
    # Initialize ROS2
    rclpy.init()
    imu_publisher = ImuPublisher()

    keyboard = KeyboardController()

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        policy = torch.jit.load(policy_path)
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        one_step_obs_size = config["one_step_obs_size"]
        obs_buffer_size = config["obs_buffer_size"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    target_dof_pos = default_angles.copy()
    action = np.zeros(num_actions, dtype=np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)
    obs_tensor_buf = torch.zeros((1, one_step_obs_size * obs_buffer_size))

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    base_body_id = 1

    # Initialize MIPO filter parameters
    mipo_config = {
        'init_cov': 0.1,
        'init_bias_cov': 1e-4,
        'init_body_height': 0.32,
        'proc_n_pos': 0.0005,
        'proc_n_vel_xy': 0.005,
        'proc_n_vel_z': 0.005,
        'proc_n_ang': 1e-7,
        'proc_n_foot_pos': 1e-4,
        'proc_n_foot_vel': 2,
        'proc_n_ba': 1e-4,
        'proc_n_bg': 1e-5,
        'proc_n_foot1_ba': 1e-4,
        'proc_n_foot2_ba': 1e-4,
        'proc_n_foot3_ba': 1e-4,
        'proc_n_foot4_ba': 1e-4,
        'ctrl_n_acc': 1e-1,
        'ctrl_n_gyro': 1e-3,
        'ctrl_n_foot1_acc': 1e-1,
        'ctrl_n_foot2_acc': 1e-1,
        'ctrl_n_foot3_acc': 1e-1,
        'ctrl_n_foot4_acc': 1e-1,
        'meas_n_zero_vel': 0.01,
        'meas_n_foot_height': 0.001,
        'mipo_use_foot_ang_contact_model': 1,
    }
    mipo = MIPOFilter(mipo_config)

    # Initialize MIPO state
    u_k_prev = None

    mujoco.mj_step(m, d)

    mocap_pos_body = d.sensordata[46:49].copy()

    # check mocap data validity
    if not np.all(mocap_pos_body == 0) and not np.any(np.isnan(mocap_pos_body)):
        init_pos = mocap_pos_body
        print(f"Using mocap initial position: {init_pos}")
    else:
        init_pos = np.array([0, 0, mipo_config.get('init_body_height', 0.32)])
        print(f"Using default initial position: {init_pos}")
    
    init_euler = np.zeros(3)
    # Transform from body frame to world frame
    R_wb = mipo.euler_to_rot(init_euler)

    actual_joint_angles = d.sensordata[:12]
    
    # Compute initial foot positions using forward kinematics
    init_foot_pos = np.zeros(12)
    for leg_id in range(4):
        angles = actual_joint_angles[leg_id*3:(leg_id+1)*3]
        foot_pos_body = mipo.forward_kinematics(angles, leg_id)
        foot_pos_world = R_wb @ foot_pos_body + init_pos
        init_foot_pos[leg_id*3:(leg_id+1)*3] = foot_pos_world
    
    print(f"Initial foot positions computed: {init_foot_pos}")
    mipo.initialize_state(init_pos, init_euler, init_foot_pos)
    
    # Lists to record MIPO estimates
    mipo_pos_list = []
    mipo_vel_list = []
    frame_pos_data_list = []
    frame_vel_data_list = []

    # Record data
    lin_vel_data_list = []
    ang_vel_data_list = []
    acc_data_list = []
    gravity_b_list = []
    joint_vel_list = []
    action_list = []
    
    # Foot IMU data
    fl_ang_vel_list = []
    fl_acc_list = []
    fr_ang_vel_list = []
    fr_acc_list = []
    rl_ang_vel_list = []
    rl_acc_list = []
    rr_ang_vel_list = []
    rr_acc_list = []

    counter = 0

    record_lists = {
        "frame_pos": frame_pos_data_list,
        "frame_vel": frame_vel_data_list,
        "ang_vel": ang_vel_data_list,
        "acc": acc_data_list,
        "gravity_b": gravity_b_list,
        "joint_vel": joint_vel_list,
        "action": action_list,
        "fl_ang_vel": fl_ang_vel_list,
        "fl_acc": fl_acc_list,
        "fr_ang_vel": fr_ang_vel_list,
        "fr_acc": fr_acc_list,
        "rl_ang_vel": rl_ang_vel_list,
        "rl_acc": rl_acc_list,
        "rr_ang_vel": rr_ang_vel_list,
        "rr_acc": rr_acc_list,
    }

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
    
            tau = pd_control(target_dof_pos, d.sensordata[:NUM_MOTOR], kps, np.zeros(12), d.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)

            d.ctrl[:] = tau

            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0 and counter > 0:
                sensors = extract_sensors(d)
                foot_forces = compute_foot_forces(m, d)

                # 獲取原始足部 IMU 測量值（在 IMU frame 中）
                gyro_feet_raw = np.concatenate([
                    sensors["fl_ang"],
                    sensors["fr_ang"],
                    sensors["rl_ang"],
                    sensors["rr_ang"],
                ])
                
                acc_feet_raw = np.concatenate([
                    sensors["fl_acc"],
                    sensors["fr_acc"],
                    sensors["rl_acc"],
                    sensors["rr_acc"],
                ])
                
                # 轉換到 body frame
                joint_angles = sensors["qpos"]
                gyro_feet_body, acc_feet_body = mipo.transform_foot_imu_to_body(
                    joint_angles, gyro_feet_raw, acc_feet_raw
                )
                
                dt = simulation_dt * control_decimation
                
                # Prepare MIPO inputs (現在使用轉換後的值)
                u_k1 = np.concatenate([
                    sensors["base_ang_vel"],  # gyro_body (已經在 body frame) (3)
                    sensors["base_acc"],       # acc_body (已經在 body frame) (3)
                    acc_feet_body,            # foot accelerations (轉換到 body frame) (12)
                    [dt]                      # hk - time step (1)
                ])  # Total: 19 dimensions (dt is also passed as separate parameter to RK4)
                
                contact_flags = (foot_forces > 5.0).astype(float)
                
                if u_k_prev is not None:
                    u_k = u_k_prev  
                else:
                    u_k = u_k1.copy()  # 第一次還是用零階保持

                # MIPO prediction and update (dt is passed as separate parameter)
                x_pred, P_pred = mipo.predict(u_k, u_k1, dt)
                
                joint_angles = sensors["qpos"]
                joint_vels = sensors["qvel"]
                
                # Extract wk (body angular velocity) and yawk (yaw from quaternion)
                wk = sensors["base_ang_vel"]  # Body angular velocity
                
                # Extract yaw from quaternion
                quat = sensors["mocap_quat_body"]  # [w, x, y, z]
                euler_from_quat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
                yawk = euler_from_quat[2]  # Yaw angle
                
                # Use transformed gyro data (in body frame)
                mipo.update(x_pred, P_pred, wk, joint_angles, joint_vels, yawk, gyro_feet_body, contact_flags)
                
                # Record MIPO estimates
                mipo_pos_list.append(mipo.x[0:3].copy())
                mipo_vel_list.append(mipo.x[3:6].copy())
                u_k_prev = u_k1.copy()
                
                publish_ros(imu_publisher, sensors, foot_forces)

                # cmd_vel = np.array(config["cmd_init"], dtype=np.float32)
                cmd_vel = keyboard.read()

                gravity_b = get_gravity_orientation(sensors["base_quat"])
                obs_list = [
                    cmd_vel * cmd_scale,
                    sensors["base_ang_vel"] * ang_vel_scale,
                    gravity_b,
                    (sensors["qpos"] - default_angles) * dof_pos_scale,
                    sensors["qvel"] * dof_vel_scale,
                    action.astype(np.float32)
                ]

                ## Record Data ##
                record_step_data(
                    record_lists,
                    sensors=sensors,
                    gravity_b=gravity_b,
                    action=action,
                    ang_vel_scale=ang_vel_scale,
                    dof_vel_scale=dof_vel_scale,
                )

                obs_list = [torch.tensor(obs, dtype=torch.float32) if isinstance(obs, np.ndarray) else obs for obs in obs_list]
                obs = torch.cat(obs_list , dim=0).unsqueeze(0)

                obs_tensor_buf = torch.cat([
                    obs,
                    obs_tensor_buf[:, : - one_step_obs_size]
                ], dim=1)
                obs_tensor = torch.clamp(obs_tensor_buf, -100, 100)

                action = policy(obs_tensor).detach().numpy().squeeze()

                # transform action to target_dof_pos
                if counter < 300:
                    target_dof_pos = default_angles
                else:
                    target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Plot the collected data after the simulation ends
    plt.figure(figsize=(14, 16))

    plt.subplot(3, 2, 1)
    for i in range(3): 
        plt.plot([step[i] for step in lin_vel_data_list], label=f"Linear Velocity {i}")
    plt.title(f"History Linear Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 2)
    for i in range(3):
        plt.plot([step[i] for step in ang_vel_data_list], label=f"Angular Velocity {i}")
    plt.title(f"History Angular Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 3)
    for i in range(3):
        plt.plot([step[i] for step in acc_data_list], label=f"Acceleration {i}")
    plt.title(f"History Acceleration", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 4)
    for i in range(3):
        plt.plot([step[i] for step in gravity_b_list], label=f"Project Gravity {i}")
    plt.title(f"History Project Gravity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 5)
    for i in range(2):
        plt.plot([step[i] for step in joint_vel_list], label=f"Joint Velocity {i}")
    plt.title(f"History Joint Velocity", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.subplot(3, 2, 6)
    for i in range(2):
        plt.plot([step[i] for step in action_list], label=f"velocity Command {i}")
    plt.title(f"History Torque Command", fontsize=10, pad=10)  # Added pad for spacing
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot foot IMU data
    plt.figure(figsize=(16, 10))
    
    # FL foot
    plt.subplot(4, 2, 1)
    for i in range(3):
        plt.plot([step[i] for step in fl_ang_vel_list], label=f"FL AngVel {['X', 'Y', 'Z'][i]}")
    plt.title("FL Foot Angular Velocity", fontsize=10, pad=10)
    plt.ylabel("rad/s")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 2, 2)
    for i in range(3):
        plt.plot([step[i] for step in fl_acc_list], label=f"FL Acc {['X', 'Y', 'Z'][i]}")
    plt.title("FL Foot Acceleration", fontsize=10, pad=10)
    plt.ylabel("m/s²")
    plt.legend()
    plt.grid(True)
    
    # FR foot
    plt.subplot(4, 2, 3)
    for i in range(3):
        plt.plot([step[i] for step in fr_ang_vel_list], label=f"FR AngVel {['X', 'Y', 'Z'][i]}")
    plt.title("FR Foot Angular Velocity", fontsize=10, pad=10)
    plt.ylabel("rad/s")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 2, 4)
    for i in range(3):
        plt.plot([step[i] for step in fr_acc_list], label=f"FR Acc {['X', 'Y', 'Z'][i]}")
    plt.title("FR Foot Acceleration", fontsize=10, pad=10)
    plt.ylabel("m/s²")
    plt.legend()
    plt.grid(True)
    
    # RL foot
    plt.subplot(4, 2, 5)
    for i in range(3):
        plt.plot([step[i] for step in rl_ang_vel_list], label=f"RL AngVel {['X', 'Y', 'Z'][i]}")
    plt.title("RL Foot Angular Velocity", fontsize=10, pad=10)
    plt.ylabel("rad/s")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 2, 6)
    for i in range(3):
        plt.plot([step[i] for step in rl_acc_list], label=f"RL Acc {['X', 'Y', 'Z'][i]}")
    plt.title("RL Foot Acceleration", fontsize=10, pad=10)
    plt.ylabel("m/s²")
    plt.legend()
    plt.grid(True)
    
    # RR foot
    plt.subplot(4, 2, 7)
    for i in range(3):
        plt.plot([step[i] for step in rr_ang_vel_list], label=f"RR AngVel {['X', 'Y', 'Z'][i]}")
    plt.title("RR Foot Angular Velocity", fontsize=10, pad=10)
    plt.ylabel("rad/s")
    plt.xlabel("Timestep")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(4, 2, 8)
    for i in range(3):
        plt.plot([step[i] for step in rr_acc_list], label=f"RR Acc {['X', 'Y', 'Z'][i]}")
    plt.title("RR Foot Acceleration", fontsize=10, pad=10)
    plt.ylabel("m/s²")
    plt.xlabel("Timestep")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    frame_pos = np.asarray(frame_pos_data_list)  # shape: (T, 3)
    frame_vel = np.asarray(frame_vel_data_list)  # shape: (T, 3)

    # Plot MIPO results
    mipo_pos_array = np.array(mipo_pos_list)
    frame_pos_array = np.array(frame_pos_data_list)
    
    # Figure 1: Position and Velocity vs Time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(mipo_pos_array[:, 0], label='MIPO X')
    plt.plot(mipo_pos_array[:, 1], label='MIPO Y')
    plt.plot(mipo_pos_array[:, 2], label='MIPO Z')
    plt.plot(frame_pos_array[:, 0], '--', label='Frame X')
    plt.plot(frame_pos_array[:, 1], '--', label='Frame Y')
    plt.plot(frame_pos_array[:, 2], '--', label='Frame Z')
    plt.title('MIPO Estimated Position')
    plt.xlabel('Time Step')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    mipo_vel_array = np.array(mipo_vel_list)
    frame_vel_array = np.array(frame_vel_data_list)
    plt.plot(mipo_vel_array[:, 0], label='MIPO Vx')
    plt.plot(mipo_vel_array[:, 1], label='MIPO Vy')
    plt.plot(mipo_vel_array[:, 2], label='MIPO Vz')
    plt.plot(frame_vel_array[:, 0], '--', label='Frame Vx')
    plt.plot(frame_vel_array[:, 1], '--', label='Frame Vy')
    plt.plot(frame_vel_array[:, 2], '--', label='Frame Vz')
    plt.title('MIPO Estimated Velocity')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Figure 2: XY Trajectory Comparison
    plt.figure(figsize=(10, 10))
    plt.plot(frame_pos_array[:, 0], frame_pos_array[:, 1], 'b-', linewidth=2, label='Ground Truth (Frame)')
    plt.plot(mipo_pos_array[:, 0], mipo_pos_array[:, 1], 'r--', linewidth=2, label='MIPO Estimate')
    plt.plot(frame_pos_array[0, 0], frame_pos_array[0, 1], 'go', markersize=10, label='Start')
    plt.plot(frame_pos_array[-1, 0], frame_pos_array[-1, 1], 'rs', markersize=10, label='End')
    plt.title('XY Trajectory Comparison', fontsize=14)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    
    plt.show()
