import pybullet as p
import pybullet_data
import time


class RobotWithGripper:
    def __init__(self, robot_urdf_path, gripper_urdf_path):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        self.robot_urdf_path = robot_urdf_path
        self.gripper_urdf_path = gripper_urdf_path
        self.load_robot()
        self.load_gripper()
        self.create_constraint()

    def load_robot(self):
        self.robot_start_pose = [0, 0, 1]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.robot_urdf_path, self.robot_start_pose, self.robot_start_orientation)
        p.loadURDF("plane.urdf")

    def load_gripper(self):
        self.gripper_start_pose = [0, 0, 1.2]
        self.gripper_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.gripper_id = p.loadURDF(self.gripper_urdf_path, self.gripper_start_pose, self.gripper_start_orientation)

    def create_constraint(self):
        # Get the number of joints in the robot to identify the end effector
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Number of joints in the robot: {num_joints}")

        # Assuming the end effector is the last joint
        end_effector_link_index = 9

        # Create a fixed constraint between the end effector of the robot and the base link of the gripper
        p.createConstraint(
            self.robot_id, end_effector_link_index,
            self.gripper_id, 0,
            p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0.1], [0, 0, -0.05]
        )

    def run_simulation(self):
        while True:
            p.stepSimulation()
            time.sleep(0.01)


if __name__ == "__main__":
    robot_urdf_path = "assets/ur5/ur5.urdf"  # Update this path
    gripper_urdf_path = "assets/gripper/robotiq_2f_85.urdf"  # Update this path
    robot_with_gripper = RobotWithGripper(robot_urdf_path, gripper_urdf_path)
    robot_with_gripper.run_simulation()
