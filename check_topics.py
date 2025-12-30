from rosbags.rosbag1 import Reader

bag_path = "/home/ray/Multi-IMU-Proprioceptive-Odometry/dataset/_2023-02-15-15-01-56.bag"

with Reader(bag_path) as reader:
    print("=== Topics in this bag ===")
    for c in reader.connections:
        print(c.topic, " | ", c.msgtype)
