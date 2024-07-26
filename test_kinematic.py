import pybullet as p
import pybullet_data
import numpy as np
import time


class PyBulletSim:
    def __init__(self, gui=True):
        self.gui = gui
        self.connect()
        self.setup_environment()
        self.load_robot()
        self.load_gripper()
        self.open_gripper()  # Đảm bảo tay kẹp mở khi mới vào

    def connect(self):
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

    def setup_environment(self):
        self.plane_id = p.loadURDF("plane.urdf")

    def load_robot(self):
        self.robot_body_id = p.loadURDF("assets/ur5/doosan_origin.urdf", [0, 0, 0.4],
                                        p.getQuaternionFromEuler([0, 0, 0]))
        self.robot_end_effector_link_index = 9
        self.robot_tool_offset = [0, 0, 0]
        self.load_robot_joints()

    def load_robot_joints(self):
        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(p.getNumJoints(self.robot_body_id))]
        self.robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self.end_effector_link_index = len(self.robot_joint_indices) - 1
        self.robot_home_joint_config = [-np.pi, -
        np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        self._joint_epsilon = 1e-2

    def load_gripper(self):
        self.gripper_urdf_path = "assets/gripper/robotiq_2f_85.urdf"
        self.gripper_start_pose = [0.5, 0.1, 0.2]
        self.gripper_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.gripper_id = p.loadURDF(self.gripper_urdf_path)
        p.resetBasePositionAndOrientation(self.gripper_id, [0, 0, 0], p.getQuaternionFromEuler([np.pi, 0, 0]))
        p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index, self.gripper_id, -1,
                           jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0], parentFramePosition=[0, 0, 0],
                           childFramePosition=self.robot_tool_offset,
                           childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]))

    def close_gripper(self):
        p.setJointMotorControl2(self.gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=10000)
        self.step_simulation(4e2)

    def open_gripper(self):
        p.setJointMotorControl2(self.gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        self.step_simulation(4e2)

    def move_joints(self, target_joint_state, speed=0.07):
        assert len(self.robot_joint_indices) == len(target_joint_state)
        p.setJointMotorControlArray(
            self.robot_body_id, self.robot_joint_indices,
            p.POSITION_CONTROL, target_joint_state,
            positionGains=speed * np.ones(len(self.robot_joint_indices))
        )
        timeout_t0 = time.time()
        while True:
            current_joint_state = [
                p.getJointState(self.robot_body_id, i)[0]
                for i in self.robot_joint_indices
            ]
            if all([
                np.abs(
                    current_joint_state[i] - target_joint_state[i]) < self._joint_epsilon
                for i in range(len(self.robot_joint_indices))
            ]):
                break
            self.step_simulation(1)

            if time.time() - timeout_t0 > 10.0:
                print(speed * np.ones(len(self.robot_joint_indices)))
                print("Timeout: robot is taking longer than 10s to reach the target joint state. Skipping...")
                break

    def move_to_points(self, points, speed):
        while True:
            for i, (target_position, target_orientation) in enumerate(points):
                target_joint_positions = p.calculateInverseKinematics(self.robot_body_id, self.end_effector_link_index,
                                                                      target_position,
                                                                      targetOrientation=target_orientation)
                self.move_joints(target_joint_positions, speed)

                # Đóng/mở tay kẹp dựa trên điểm mục tiêu
                if i % 2 == 0:
                    self.close_gripper()  # Đóng tay kẹp ở các điểm 0, 2, 4, ...
                else:
                    self.open_gripper()  # Mở tay kẹp ở các điểm 1, 3, 5, ...

                time.sleep(1)  # Đợi một chút trước khi di chuyển đến điểm tiếp theo

    def step_simulation(self, num_steps):
        for i in range(int(num_steps)):
            p.stepSimulation()
            if hasattr(self, 'gripper_id') and self.gripper_id is not None:
                # Constraints
                gripper_joint_positions = np.array(
                    [p.getJointState(self.gripper_id, i)[0] for i in range(p.getNumJoints(self.gripper_id))])
                p.setJointMotorControlArray(self.gripper_id, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
                                            [gripper_joint_positions[1], -gripper_joint_positions[1],
                                             -gripper_joint_positions[1], gripper_joint_positions[1],
                                             gripper_joint_positions[1]],
                                            positionGains=np.ones(5))
            time.sleep(1e-3)

    def add_objects(self):
        # Thêm các vật thể vào môi trường với tọa độ và hướng cụ thể
        self.object1_id = p.loadURDF("assets/objects/cube.urdf", [0.5, 0, 0])
        self.object2_id = p.loadURDF("assets/objects/rod.urdf", [-0.5, 0, 0])
def main():
    sim = PyBulletSim(gui=True)
    sim.add_objects()
    # Đặt ba điểm mục tiêu
    points = [
        ([2, 0, 0], p.getQuaternionFromEuler([0, np.pi, 0])),
        ([0, 0, 5], p.getQuaternionFromEuler([0, 0, np.pi//2])),
        ([-2, 0, 0], p.getQuaternionFromEuler([0, np.pi, 0])),
    ]

    # Thời gian di chuyển giữa các điểm (s)
    speed = 0.005

    # Di chuyển robot qua ba điểm
    sim.move_to_points(points, speed)


if __name__ == "__main__":
    main()
