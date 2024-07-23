import pybullet as p
import pybullet_data
import numpy as np
import time


def move_joints_smoothly(robot_body_id, joint_indices, target_joint_positions, duration):
    """
        Move robot joints to target positions smoothly over a given duration.
    """
    start_joint_positions = [p.getJointState(robot_body_id, i)[0] for i in joint_indices]
    t0 = time.time()
    while True:
        elapsed_time = time.time() - t0
        alpha = min(elapsed_time / duration, 1.0)  # Calculate interpolation factor
        new_joint_positions = [
            start + alpha * (target - start)
            for start, target in zip(start_joint_positions, target_joint_positions)
        ]
        p.setJointMotorControlArray(
            robot_body_id, joint_indices,
            p.POSITION_CONTROL, new_joint_positions,
            positionGains=np.ones(len(joint_indices))
        )

        if alpha >= 1.0:
            break

        p.stepSimulation()
        time.sleep(1. / 240.)


def main():
    # Kết nối PyBullet với GUI
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Tải môi trường và robot
    plane_id = p.loadURDF("plane.urdf")
    robot_body_id = p.loadURDF("assets/ur5/doosan_origin.urdf", [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))

    # Lấy thông tin khớp robot
    robot_joint_info = [p.getJointInfo(robot_body_id, i) for i in range(p.getNumJoints(robot_body_id))]
    robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
    endEffectorLinkIndex = len(robot_joint_indices) - 1  # Link cuối cùng

    # Đặt ba điểm mục tiêu
    points = [
        ([2, 0.5, 0.5], p.getQuaternionFromEuler([0, 0, 0])),
        ([0.5, -0.5, 0.5], p.getQuaternionFromEuler([0, np.pi / 2, 0])),
        ([-0.5, -0.5, 0.5], p.getQuaternionFromEuler([0, np.pi, 0])),
    ]

    # Thời gian di chuyển giữa các điểm (s)
    move_duration = 1.0

    # Vòng lặp chính để di chuyển robot qua ba điểm
    while True:
        for target_position, target_orientation in points:
            # Tính toán các góc khớp mục tiêu
            target_joint_positions = p.calculateInverseKinematics(
                robot_body_id,
                endEffectorLinkIndex,
                target_position,
                targetOrientation=target_orientation
            )

            # Di chuyển các khớp một cách mượt mà đến các vị trí mục tiêu
            move_joints_smoothly(robot_body_id, robot_joint_indices, target_joint_positions, move_duration)

        # Lặp lại liên tục
        time.sleep(1.0)


if __name__ == "__main__":
    main()
