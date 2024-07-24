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
        self.add_objects()  # Thêm các vật thể vào môi trường

    def connect(self):
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

    def setup_environment(self):
        self.plane_id = p.loadURDF("plane.urdf")

    def load_robot(self):
        self.robot_body_id = p.loadURDF("assets/ur5/doosan_origin.urdf", [0, 0, 0.4],
                                        p.getQuaternionFromEuler([0, 0, 0]))
        self.robot_end_effector_link_index = 7
        self.robot_tool_offset = [0, 0, 0]
        self.load_robot_joints()

    def load_robot_joints(self):
        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(p.getNumJoints(self.robot_body_id))]
        self.robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self.end_effector_link_index = len(self.robot_joint_indices) - 1

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

    def add_objects(self):
        # Thêm các vật thể vào môi trường với tọa độ và hướng cụ thể
        self.object1_id = p.loadURDF("assets/objects/cube.urdf", [0.5, 0, 0])
        self.object2_id = p.loadURDF("assets/objects/rod.urdf", [-0.5, 0, 0])


    def close_gripper(self):
        p.setJointMotorControl2(self.gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=10000)
        self.step_simulation(4e2)

    def open_gripper(self):
        p.setJointMotorControl2(self.gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        self.step_simulation(4e2)

    def move_joints_smoothly(self, target_joint_positions, duration):
        start_joint_positions = [p.getJointState(self.robot_body_id, i)[0] for i in self.robot_joint_indices]
        t0 = time.time()
        while True:
            elapsed_time = time.time() - t0
            alpha = min(elapsed_time / duration, 1.0)
            new_joint_positions = [start + alpha * (target - start) for start, target in
                                   zip(start_joint_positions, target_joint_positions)]
            p.setJointMotorControlArray(self.robot_body_id, self.robot_joint_indices, p.POSITION_CONTROL,
                                        new_joint_positions,
                                        positionGains=np.ones(len(self.robot_joint_indices)))
            if alpha >= 1.0:
                break
            p.stepSimulation()
            time.sleep(1. / 240.)

    def move_to_object(self, object_id, move_duration):
        # Lấy vị trí của vật thể
        object_position, _ = p.getBasePositionAndOrientation(object_id)

        # Tính toán các góc khớp mục tiêu để robot tiếp cận vật thể
        target_joint_positions = p.calculateInverseKinematics(
            self.robot_body_id, self.end_effector_link_index, object_position
        )

        # Di chuyển robot đến vị trí của vật thể
        self.move_joints_smoothly(target_joint_positions, move_duration)

        # Đóng tay kẹp để gắp vật thể
        self.close_gripper()

        # Kiểm tra việc gắp
        time.sleep(2)  # Đợi một chút để đảm bảo tay kẹp đã gắp vật thể

        # Mở tay kẹp nếu cần thiết (tuỳ theo nhu cầu)
        # self.open_gripper()

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


def main():

    sim = PyBulletSim(gui=True)

    # Di chuyển robot đến các vật thể và gắp chúng
    object_ids = [sim.object1_id, sim.object2_id]
    move_duration = 2  # Thời gian di chuyển đến vị trí của vật thể

    while True:
        for object_id in object_ids:
            sim.move_to_object(object_id, move_duration)
            # Bạn có thể thêm logic để di chuyển vật thể đến vị trí khác hoặc thực hiện các hành động khác
            time.sleep(1.0)  # Đợi một chút trước khi di chuyển đến vật thể tiếp theo


if __name__ == "__main__":
    main()
