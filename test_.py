import pybullet as p
import pybullet_data
import numpy as np
import time

class PyBulletSim:
    def __init__(self, gui=True):
        # load environment
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        self.robot_body_id = p.loadURDF(
            "assets/ur5/doosan_origin.urdf", [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))
        self.robot_end_effector_link_index = 7
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])

        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(
            p.getNumJoints(self.robot_body_id))]
        self._robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]

        self._joint_epsilon = 1e-3

        self.robot_home_joint_config = [-np.pi, -
                                        np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        self.robot_goal_joint_config = [
            0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

        self._gripper_body_id = None  # Initialize gripper body ID

        self.move_joints(self.robot_home_joint_config)

    def load_gripper(self):
        if self._gripper_body_id is not None:
            print("Gripper already loaded")
            return

        self._gripper_body_id = p.loadURDF("assets/gripper/robotiq_2f_85.urdf")

        p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index, self._gripper_body_id, -1, jointType=p.JOINT_FIXED, jointAxis=[
                           0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0], childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        for i in range(p.getNumJoints(self._gripper_body_id)):
            p.changeDynamics(self._gripper_body_id, i, lateralFriction=1.0, spinningFriction=1.0,
                             rollingFriction=0.0001, frictionAnchor=True)
        self.step_simulation(100)

    def move_joints(self, target_joint_state, speed=0.03):
        assert len(self._robot_joint_indices) == len(target_joint_state)
        p.setJointMotorControlArray(
            self.robot_body_id, self._robot_joint_indices,
            p.POSITION_CONTROL, target_joint_state,
            positionGains=speed * np.ones(len(self._robot_joint_indices))
        )

        timeout_t0 = time.time()
        while True:
            current_joint_state = [
                p.getJointState(self.robot_body_id, i)[0]
                for i in self._robot_joint_indices
            ]
            if all([
                np.abs(
                    current_joint_state[i] - target_joint_state[i]) < self._joint_epsilon
                for i in range(len(self._robot_joint_indices))
            ]):
                break
            self.step_simulation(1)

            if time.time() - timeout_t0 > 10.0:
                print("Timeout: robot is taking longer than 10s to reach the target joint state. Skipping...")
                break

    def move_tool(self, tool_pose):
        tool_position = tool_pose[:3]
        tool_rotation = tool_pose[3]
        target_position = tool_position - self._tool_tip_to_ee_joint

        target_joint_state = p.calculateInverseKinematics(
            self.robot_body_id, self.robot_end_effector_link_index, target_position,
            targetOrientation=p.getQuaternionFromEuler(
                [np.pi, 0, tool_rotation]),
            lowerLimits=[-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -2*np.pi],
            upperLimits=[np.pi, 0, np.pi, 0, np.pi, 2*np.pi],
            jointRanges=[2*np.pi]*6,
            restPoses=self.robot_goal_joint_config
        )
        self.move_joints(target_joint_state)

    def control_gripper(self, action):
        if action == "open":
            target_joint_state = [0.04, 0.04]
        elif action == "close":
            target_joint_state = [0.00, 0.00]
        else:
            raise ValueError("Invalid action for gripper control")

        gripper_joint_indices = [1, 3]
        p.setJointMotorControlArray(
            self._gripper_body_id, gripper_joint_indices,
            p.POSITION_CONTROL, target_joint_state,
            positionGains=np.ones(len(gripper_joint_indices))
        )
        timeout_t0 = time.time()
        while True:
            current_joint_state = [
                p.getJointState(self._gripper_body_id, i)[0]
                for i in gripper_joint_indices
            ]
            if all([
                np.abs(current_joint_state[i] - target_joint_state[i]) < self._joint_epsilon
                for i in range(len(gripper_joint_indices))
            ]):
                break
            self.step_simulation(1)
            if time.time() - timeout_t0 > 10.0:
                print("Timeout: gripper is taking longer than 10s to reach the target joint state. Skipping...")
                break

    def step_simulation(self, num_steps):
        for _ in range(int(num_steps)):
            p.stepSimulation()

if __name__ == "__main__":
    sim = PyBulletSim(gui=True)
    sim.load_gripper()
    sim.move_tool([0.5, 0.0, 0.1, np.pi/4])
    sim.control_gripper("close")
    time.sleep(1)
    sim.control_gripper("open")
    time.sleep(1)
    p.disconnect()
