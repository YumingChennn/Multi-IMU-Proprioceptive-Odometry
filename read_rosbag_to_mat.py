from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_ros1

import numpy as np
import scipy.io as sio

bag_path = "/home/ray/Multi-IMU-Proprioceptive-Odometry/dataset/_2023-02-15-15-01-56.bag"

# topics from MATLAB
topic_fl = "/WT901_49_Data"
topic_fr = "/WT901_48_Data"
topic_rl = "/WT901_50_Data"
topic_rr = "/WT901_47_Data"

topic_body = "/unitree_hardware/imu"
topic_mocap_body = "/mocap_node/Go1_body/pose"
topic_mocap_fr = "/mocap_node/Go1_FR/pose"
topic_joint = "/unitree_hardware/joint_foot"

# topics we choose to ignore (images too large)
ignored_topics = {
    "/camera_forward/infra1/image_rect_raw",
    "/camera_forward/infra2/image_rect_raw",
}

def read_topic(topic):
    """Generic reader for hybrid ROS1 bag with ROS2 msg types."""
    t_list = []
    data_list = []

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():

            if connection.topic in ignored_topics:
                continue

            if connection.topic != topic:
                continue

            msg = deserialize_ros1(rawdata, connection.msgtype)

            # ============== IMU ==============
            if connection.msgtype == "sensor_msgs/msg/Imu":
                data_list.append([
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ])

            # ============== Mocap PoseStamped ==============
            elif connection.msgtype == "geometry_msgs/msg/PoseStamped":
                data_list.append([
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                    msg.pose.orientation.w,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                ])

            # ============== Joint Foot ==============
            elif connection.msgtype == "sensor_msgs/msg/JointState":

                # Debug print for FIRST message only
                if len(data_list) == 0:
                    print("\n===== JointState Structure Detected =====")
                    print("Length of position  :", len(msg.position))
                    print("Length of velocity  :", len(msg.velocity))
                    print("Length of effort    :", len(msg.effort))

                    print("\nSample position:", msg.position[:])
                    print("Sample velocity:", msg.velocity[:])
                    print("Sample effort  :", msg.effort[:])
                    print("========================================\n")

                data_list.append(
                    list(msg.position) +
                    list(msg.velocity) +
                    list(msg.effort)
                )

            else:
                continue

            t_list.append(timestamp * 1e-9)  # convert ns → sec

    return np.array(t_list), np.array(data_list)


# ========== BEGIN READING ==========

print("Reading FL IMU...")
t_fl, imu_fl = read_topic(topic_fl)

print("Reading FR IMU...")
t_fr, imu_fr = read_topic(topic_fr)

print("Reading RL IMU...")
t_rl, imu_rl = read_topic(topic_rl)

print("Reading RR IMU...")
t_rr, imu_rr = read_topic(topic_rr)

print("Reading body IMU...")
t_body, imu_body = read_topic(topic_body)

print("Reading mocap (body)...")
t_mocap_body, mocap_body = read_topic(topic_mocap_body)

print("Reading mocap (FR foot)...")
t_mocap_fr, mocap_fr = read_topic(topic_mocap_fr)

print("Reading joint foot...")
t_joint, joint_data = read_topic(topic_joint)


def debug_show(name, t, data):
    print("\n==============================")
    print(f"Topic: {name}")
    print("==============================")
    print("Data shape:", data.shape)
    print("Timestamp shape:", t.shape)

    if data.size == 0:
        print("⚠️  No data in this topic!")
        return

    print("\nFirst 5 timestamps:")
    print(t[:5])

    print("\nFirst 5 data rows:")
    print(data[:5])

    print("\nValue ranges:")
    print("  min:", np.min(data, axis=0))
    print("  max:", np.max(data, axis=0))


# Debug all topics
debug_show("imu_fl", t_fl, imu_fl)
debug_show("imu_fr", t_fr, imu_fr)
debug_show("imu_rl", t_rl, imu_rl)
debug_show("imu_rr", t_rr, imu_rr)
debug_show("imu_body", t_body, imu_body)

debug_show("mocap_body", t_mocap_body, mocap_body)
debug_show("mocap_FR", t_mocap_fr, mocap_fr)

debug_show("joint_foot", t_joint, joint_data)


# ========== OPTIONAL: SAVE TO MAT FILE FOR MATLAB ==========
print("Saving to bag_data.mat ...")

sio.savemat("bag_data.mat", {
    "imu_fl_time": t_fl,
    "imu_fl_data": imu_fl,

    "imu_fr_time": t_fr,
    "imu_fr_data": imu_fr,

    "imu_rl_time": t_rl,
    "imu_rl_data": imu_rl,

    "imu_rr_time": t_rr,
    "imu_rr_data": imu_rr,

    "imu_body_time": t_body,
    "imu_body_data": imu_body,

    "mocap_body_time": t_mocap_body,
    "mocap_body_data": mocap_body,

    "mocap_FR_time": t_mocap_fr,
    "mocap_FR_data": mocap_fr,

    "joint_time": t_joint,
    "joint_data": joint_data,
})

print("✔ DONE — bag_data.mat generated successfully!")
