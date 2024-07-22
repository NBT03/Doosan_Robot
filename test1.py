import pybullet as p
import pybullet_data
import numpy as np
import sys

# Khởi tạo môi trường PyBullet
physicsClient = p.connect(p.GUI)  # Mở môi trường với GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, 0)

# Kiểm tra xem có đường dẫn được cung cấp qua dòng lệnh không
# if len(sys.argv) < 2:
#     print("Cách sử dụng: python3 test_urdf.py <đường dẫn tới file URDF>")
#     exit(1)

# Lấy đường dẫn file URDF từ dòng lệnh
# urdf_path = sys.argv[1]

# Load robot từ file URDF
robot_urdf_path = "assets/ur5/doosan_origin.urdf"  # Đường dẫn tới file URDF của bạn
robot_start_pose = [0, 0, 0]  # Vị trí ban đầu của robot
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Hướng ban đầu của robot (ở đây là không xoay)
robot_id = p.loadURDF(robot_urdf_path, robot_start_pose, robot_start_orientation)
gripper_urdf_path = "assets/gripper/robotiq_2f_85.urdf"
gripper_start_pose = [0.5, 0.1, 0.2]
gripper_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
gripper_id = p.loadURDF(gripper_urdf_path)
robot_end_effector_link_index = 5
robot_tool_offset = [0, 0, 0]
p.resetBasePositionAndOrientation(gripper_id, [
                                     0, 0, 0], p.getQuaternionFromEuler([np.pi, 0, 0]))
p.createConstraint(robot_id, robot_end_effector_link_index, gripper_id, -1,
                           jointType=p.JOINT_FIXED, jointAxis=[
                0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=robot_tool_offset,
                           childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]))

# Load sàn hoặc giá để đặt robot lên đó
plane_id = p.loadURDF("plane.urdf")  # Load sàn hoặc giá
p.createConstraint(
                robot_id, -1, -1, -1, p.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, 0.4])

while True:
    p.stepSimulation()
