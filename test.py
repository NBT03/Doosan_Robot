import pybullet as p
import pybullet_data
import numpy as np
import time


def move_joints(robot_body_id, joint_indices, start_joint_state, end_joint_state, duration, speed=0.03):
    """
        Move robot arm from start_joint_state to end_joint_state over a given duration.
    """
    num_steps = int(duration / (1.0 / 240.0))  # 240 Hz simulation step rate
    step_interval = duration / num_steps
    joint_states = np.linspace(start_joint_state, end_joint_state, num_steps)

    for state in joint_states:
        p.setJointMotorControlArray(
            robot_body_id, joint_indices,
            p.POSITION_CONTROL, state,
            positionGains=speed * np.ones(len(joint_indices))
        )
        p.stepSimulation()
        time.sleep(step_interval)


def main():
    # Kết nối PyBullet với GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Tải môi trường và robot
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("assets/ur5/doosan_origin.urdf", [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))
    gripper_urdf_path = "assets/gripper/robotiq_2f_85.urdf"
    gripper_start_pose = [0.5, 0.1, 0.2]
    gripper_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    gripper_id = p.loadURDF(gripper_urdf_path)
    robot_end_effector_link_index = 7
    robot_tool_offset = [0, 0, 0]
    p.resetBasePositionAndOrientation(gripper_id, [
        0, 0, 0], p.getQuaternionFromEuler([np.pi, 0, 0]))
    p.createConstraint(robot_id, robot_end_effector_link_index, gripper_id, -1,
                       jointType=p.JOINT_FIXED, jointAxis=[
            0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=robot_tool_offset,
                       childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]))

    # Lấy thông tin khớp robot
    robot_joint_info = [p.getJointInfo(robot_id, i) for i in range(p.getNumJoints(robot_id))]
    robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]

    # Cấu hình khớp ban đầu và đích
    robot_home_joint_config = [-np.pi / 2, 0, np.pi / 4, -np.pi / 4, np.pi / 3, 0]
    robot_goal_joint_config = [np.pi / 3, -np.pi / 3, -np.pi / 4, np.pi / 4, -np.pi / 6, np.pi / 2]

    # Vòng lặp chính để di chuyển giữa hai cấu hình
    while True:
        # Di chuyển khớp từ cấu hình ban đầu đến cấu hình đích
        move_joints(robot_id, robot_joint_indices, robot_home_joint_config, robot_goal_joint_config, duration=5.0,
                    speed=0.03)

        # Tạm dừng để quan sát
        time.sleep(2)

        # Di chuyển khớp từ cấu hình đích trở về cấu hình ban đầu
        move_joints(robot_id, robot_joint_indices, robot_goal_joint_config, robot_home_joint_config, duration=5.0,
                    speed=0.03)

        # Tạm dừng để quan sát
        time.sleep(2)


if __name__ == "__main__":
    main()
