import pybullet as p
import time
import pybullet_data
import math
from collections import namedtuple
from attrdict import AttrDict

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, 0)

p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.5, -0.9, 0.5])

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("assets/ur5/doosan.urdf", useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
numJoints = p.getNumJoints(robotId)
jointInfo = namedtuple("jointInfo",
                       ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
joints = AttrDict()
for i in range(numJoints):
    info = p.getJointInfo(robotId, i)
    print(info)
    jointID = info[0]
    jointName = info[1].decode('utf-8')
    jointType = jointTypeList[info[2]]
    jointLowerLimit = info[8]
    jointUpperLimit = info[9]
    jointMaxForce = info[10]
    jointMaxVelocity = info[11]
    singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                           jointMaxVelocity, True)
    joints[singleInfo.name] = singleInfo

print(joints)

for jointName in joints:
    print("jointName:", jointName)

position_control_group = []
position_control_group.append(p.addUserDebugParameter('joint1', -math.pi, math.pi, 0))
position_control_group.append(p.addUserDebugParameter('joint2', -math.pi, math.pi, 0))
position_control_group.append(p.addUserDebugParameter('joint3', -0.5 * math.pi, 0.5 * math.pi, 0))
position_control_group.append(p.addUserDebugParameter('joint4', -math.pi, math.pi, 0))
position_control_group.append(p.addUserDebugParameter('joint5', -0.5 * math.pi, 0.5 * math.pi, 0))
position_control_group.append(p.addUserDebugParameter('joint6', -math.pi, math.pi, 0))

position_control_joint_name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
print("position_control_group:", position_control_group)
while True:
    time.sleep(1 / 240.)
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

    parameter = []
    for i in range(6):
        parameter.append(p.readUserDebugParameter(position_control_group[i]))
    num = 0
    # print("parameter:",parameter)
    for jointName in joints:
        if jointName in position_control_joint_name:
            joint = joints[jointName]
            parameter_sim = parameter[num]
            p.setJointMotorControl2(robotId, joint.id, p.POSITION_CONTROL,
                                    targetPosition=parameter_sim,
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)
            num = num + 1
    p.stepSimulation()