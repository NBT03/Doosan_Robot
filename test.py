import pybullet as p
import pybullet_data
import numpy as np
import time


class PyBulletSim:
    """
    PyBulletSim: Implements two tote UR5 simulation environment with obstacles for grasping
        and manipulation
    """

    def __init__(self, use_random_objects=False, object_shapes=None, gui=True):
        # 3D workspace for tote 1
        self._workspace1_bounds = np.array([
            [0.38, 0.62],  # 3x2 rows: x,y,z cols: min,max
            [-0.22, 0.22],
            [0.00, 0.5]
        ])
        # 3D workspace for tote 2
        self._workspace2_bounds = np.copy(self._workspace1_bounds)
        self._workspace2_bounds[0, :] = - self._workspace2_bounds[0, ::-1]

        # load environment
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        # load UR5 robot
        self.robot_body_id = p.loadURDF(
            "assets/ur5/ur5.urdf", [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]))
        self._mount_body_id = p.loadURDF(
            "assets/ur5/mount.urdf", [0, 0, 0.2], p.getQuaternionFromEuler([0, 0, 0]))

        # Placeholder for gripper body id
        self._gripper_body_id = None
        self.robot_end_effector_link_index = 9
        self._robot_tool_offset = [0, 0, -0.05]
        # Distance between tool tip and end-effector joint
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])

        # Get revolute joint indices of robot (skip fixed joints)
        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(
            p.getNumJoints(self.robot_body_id))]
        self._robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]

        # joint position threshold in radians (i.e. move until joint difference < epsilon)
        self._joint_epsilon = 1e-3

        # Robot home joint configuration (over tote 1)
        self.robot_home_joint_config = [-np.pi, -
                                        np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        # Robot goal joint configuration (over tote 2)
        self.robot_goal_joint_config = [
            0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]

        self.move_joints(self.robot_home_joint_config, speed=1.0)

        # Load totes and fix them to their position
        self._tote1_position = (
            self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
        self._tote1_position[2] = 0.01
        self._tote1_body_id = p.loadURDF(
            "assets/tote/toteA_large.urdf", self._tote1_position, p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True)

        self._tote2_position = (
            self._workspace2_bounds[:, 0] + self._workspace2_bounds[:, 1]) / 2
        self._tote2_position[2] = 0.01
        self._tote2_body_id = p.loadURDF(
            "assets/tote/toteA_large.urdf", self._tote2_position, p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True)

        # Load objects
        # - possible object colors
        self._object_colors = get_tableau_palette()

        # - Define possible object shapes
        if object_shapes is not None:
            self._object_shapes = object_shapes
        else:
            self._object_shapes = [
                "assets/objects/cube.urdf",
                "assets/objects/rod.urdf",
                "assets/objects/custom.urdf",
            ]
        self._num_objects = len(self._object_shapes)
        self._object_shape_ids = [
            i % len(self._object_shapes) for i in range(self._num_objects)]
        self._objects_body_ids = []
        for i in range(self._num_objects):
            object_body_id = p.loadURDF(self._object_shapes[i], [0.5, 0.1, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
            self._objects_body_ids.append(object_body_id)
            p.changeVisualShape(object_body_id, -1, rgbaColor=[*self._object_colors[i], 1])
        self.reset_objects()

        # Add obstacles
        self.obstacles = [
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, 0.65, 0.9],
                       useFixedBase=True
                       ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, 0.65, 0.3],
                       useFixedBase=True
                       ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, -0.65, 0.9],
                       useFixedBase=True
                       ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, -0.65, 0.3],
                       useFixedBase=True
                       ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0, 0, 1.5],
                       useFixedBase=True
                       ),
        ]
        self.obstacles.extend(
            [self._plane_id, self._tote1_body_id, self._tote2_body_id])

    def load_gripper(self):
        if self._gripper_body_id is not None:
            print("Gripper already loaded")
            return

        # Attach robotiq gripper to UR5 robot
        # - We use createConstraint to add a fixed constraint between the ur5 robot and gripper.
        self._gripper_body_id = p.loadURDF("assets/gripper/robotiq_2f_85.urdf")
        p.resetBasePositionAndOrientation(self._gripper_body_id, [
            0.5, 0.1, 0.2], p.getQuaternionFromEuler([np.pi, 0, 0]))

        p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index, self._gripper_body_id, 0,
                           jointType=p.JOINT_FIXED, jointAxis=[
                               0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset,
                           childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self._gripper_body_id)):
            p.changeDynamics(self._gripper_body_id, i, lateralFriction=1.0, spinningFriction=1.0,
                             rollingFriction=0.0001, frictionAnchor=True)
        self.step_simulation(1e3)

    def move_joints(self, target_joint_state, speed=0.03):
        """
            Move robot arm to specified joint configuration by appropriate motor control
        """
        assert len(self._robot_joint_indices) == len(target_joint_state)
        p.setJointMotorControlArray(
            self.robot_body_id, self._robot_joint_indices,
            p.POSITION_CONTROL, target_joint_state,
            positionGains=speed * np.ones(len(self._robot_joint_indices))
        )

        timeout_t0 = time.time()
        while True:
            # Keep moving until joints reach the target configuration
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
            if time.time() - timeout_t0 > 10:
                print(
                    "Timeout: robot is taking longer than 10s to reach the target joint state. Skipping...")
                p.setJointMotorControlArray(
                    self.robot_body_id, self._robot_joint_indices,
                    p.POSITION_CONTROL, self.robot_home_joint_config,
                    positionGains=np.ones(len(self._robot_joint_indices))
                )
                break
            self.step_simulation(1)

    def move_to(self, pos, orn=None, tolerance=0.01, timeout=3):
        """
            Move gripper to target position using inverse kinematics (IK)
        """
        joint_configs = p.calculateInverseKinematics(
            self.robot_body_id, self.robot_end_effector_link_index, pos, orn)
        self.move_joints(joint_configs, speed=1.0)

    def reset_objects(self):
        for obj_body_id in self._objects_body_ids:
            pos = [
                np.random.uniform(self._workspace1_bounds[i, 0], self._workspace1_bounds[i, 1])
                for i in range(3)
            ]
            orn = [0, 0, np.random.uniform(0, 2 * np.pi)]
            p.resetBasePositionAndOrientation(obj_body_id, pos, p.getQuaternionFromEuler(orn))

    def open_gripper(self, open_width=0.08, step_sim=True):
        """
            Open robotiq gripper to specified width
        """
        if self._gripper_body_id is None:
            print("Error: no gripper loaded")
            return
        open_pos = open_width / 2
        p.setJointMotorControl2(
            self._gripper_body_id, 1, p.POSITION_CONTROL, targetPosition=open_pos, force=100)
        p.setJointMotorControl2(
            self._gripper_body_id, 3, p.POSITION_CONTROL, targetPosition=open_pos, force=100)
        p.setJointMotorControl2(
            self._gripper_body_id, 4, p.POSITION_CONTROL, targetPosition=-open_pos, force=100)
        p.setJointMotorControl2(
            self._gripper_body_id, 6, p.POSITION_CONTROL, targetPosition=-open_pos, force=100)
        if step_sim:
            self.step_simulation(200)

    def close_gripper(self, close_force=100, step_sim=True):
        """
            Close robotiq gripper
        """
        if self._gripper_body_id is None:
            print("Error: no gripper loaded")
            return
        p.setJointMotorControl2(
            self._gripper_body_id, 1, p.POSITION_CONTROL, targetPosition=0, force=close_force)
        p.setJointMotorControl2(
            self._gripper_body_id, 3, p.POSITION_CONTROL, targetPosition=0, force=close_force)
        p.setJointMotorControl2(
            self._gripper_body_id, 4, p.POSITION_CONTROL, targetPosition=0, force=close_force)
        p.setJointMotorControl2(
            self._gripper_body_id, 6, p.POSITION_CONTROL, targetPosition=0, force=close_force)
        if step_sim:
            self.step_simulation(200)

    def step_simulation(self, step_count):
        for _ in range(step_count):
            p.stepSimulation()


# Utility functions
def get_tableau_palette():
    """
    returns a beautiful color palette
    :return palette (np.array object): np array of rgb colors in range [0, 1]
    """
    palette = np.array(
        [
            [89, 169, 79],  # green
            [156, 117, 95],  # brown
            [237, 201, 72],  # yellow
            [78, 121, 167],  # blue
            [255, 87, 89],  # red
            [242, 142, 43],  # orange
            [176, 122, 161],  # purple
            [255, 157, 167],  # pink
            [118, 183, 178],  # cyan
            [186, 176, 172]  # gray
        ],
        dtype=np.cfloat
    )
if __name__ == '__main__':
    sim = PyBulletSim()
    sim.load_gripper()

    object_pos = [0.5, 0, 0.8]
    sim.move_to(object_pos)
    sim.close_gripper()

    lift_pos = [0.5, 0, 1.0]
    sim.move_to(lift_pos)
    sim.open_gripper()

    p.disconnect()
