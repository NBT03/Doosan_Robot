import sim
from random import seed
import os
import camera
import pybullet as p
import numpy as np
import image
import torch
import train_seg_model
from PIL import Image
import torchvision
import icp
import transforms
from scipy.spatial.transform import Rotation
import transforms3d
import random
import main
from train_seg_model import RGBDataset

if __name__ == "__main__":
    random.seed(1)
    color_palette = train_seg_model.get_tableau_palette()

    # Note: Please don't change the order in object_shapes and object_meshes array.
    #   their order is consistent with the trained segmentation model.
    object_shapes = [
        "assets/objects/cube.urdf",
        "assets/objects/rod.urdf",
        "assets/objects/custom.urdf",
    ]
    object_meshes = [
        "assets/objects/cube.obj",
        "assets/objects/rod.obj",
        "assets/objects/custom.obj",
    ]
    env = sim.PyBulletSim(object_shapes=object_shapes)
    env.load_gripper()

    # Predefined grasping transformation wrt each object, i.e.,
    #  when an object i loaded at basePosition [0,0,0] and baseOrientation [0,0,0]
    #  is attempted to grasp at corresponding object_grasp_positions[i] and
    #  object_grasp_angles[i], it will result in a successful grasp.
    object_grasp_positions = [
        np.array([[0, 0, 0, 1]]).transpose(),
        np.array([[0, 0, 0, 1]]).transpose(),
        np.array([[-1.96, 5, 0, 100]]).transpose() / 100
    ]
    object_grasp_angles = [0, 0, 0]

    # Setup camera (this should be consistent with the camera
    # used during training segmentation model)
    my_camera = camera.Camera(
        image_size=(480, 640),
        near=0.01,
        far=10.0,
        fov_w=50
    )
    camera_target_position = (env._workspace1_bounds[:, 0] + env._workspace1_bounds[:, 1]) / 2
    camera_target_position[2] = 0
    camera_distance = np.sqrt(((np.array([0.5, -0.5, 0.8]) - camera_target_position)**2).sum())
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target_position,
        distance=camera_distance,
        yaw=90,
        pitch=-60,
        roll=0,
        upAxisIndex=2,
    )

    # Prepare model (again, should be consistent with segmentation training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = 3  # RGB
    n_classes = len(object_shapes) + 1  # number of objects + 1 for background class
    model = train_seg_model.miniUNet(n_channels, n_classes)
    model.to(device)
    model, _, _ = train_seg_model.load_chkpt(model, 'checkpoint.pth.tar', device)
    model.eval()

    obj_ids = env._objects_body_ids  # everything else will be treated as background

    is_grasped = np.zeros(3).astype(np.bool_)
    while not np.all(is_grasped):  # Keep repeating until the tote is empty
        # Capture rgb and depth image of the tote.
        rgb_obs, depth_obs, _ = camera.make_obs(my_camera, view_matrix)

        # Generate the segmentation prediction from the model
        dataset = RGBDataset("")
        rgb_img = dataset.transform(rgb_obs)
        rgb_img = rgb_img.reshape([1, 3, 480, 640])
        rgb_img = rgb_img.to(device)

        output = model(rgb_img)
        _, pred = torch.max(output, dim=1)
        pred = np.squeeze(pred)  # pred[480x640]
        pred_mask = pred.cpu().numpy()

        markers = []
        # Points in each point cloud to use for ICP.
        # Adjust this as per your machine performance.
        num_sample_pts = 500

        # Randomly choose an object index to grasp which is not grasped yet.
        obj_id = np.random.choice(np.where(~is_grasped)[0], 1)[0] + 1  # objects ID start from 1 to 3
        obj_index = obj_id - 1

        # Mask out the depth based on predicted segmentation mask of object.
        obj_depth = icp.gen_obj_depth(obj_id, depth_obs, pred_mask)

        # Transform depth to 3d points in camera frame.
        cam_pts = np.zeros((0, 3))
        cam_pts = np.array(transforms.depth_to_point_cloud(my_camera.intrinsic_matrix, obj_depth))
        if cam_pts.shape == (0,):
            print("No points are present in segmented point cloud. Please check your code. Continuing ...")
            continue

        # Transform 3d points (seg_pt_cloud) in camera frame to the world frame
        camera_pose = camera.cam_view2pose(view_matrix)
        world_pts = transforms.transform_point3s(camera_pose, cam_pts)

        world_pts_sample = world_pts[np.random.choice(range(world_pts.shape[0]), num_sample_pts), :]
        mesh_path = object_meshes[obj_index]  # object index starts from 0 to 2
        gt_pts_sample = icp.mesh2pts(mesh_path, N=len(world_pts_sample))

        # Align ground truth point cloud (gt_pt_cloud) to segmented point cloud (seg_pt_cloud) using ICP.
        transform, transformed = icp.align_pts(gt_pts_sample, world_pts_sample, max_iterations=200, threshold=1e-7)

        # Transform pre-defined grasp position `obj_grasp_positions[obj_index]` from ground truth object frame to the segmented object frame.
        position = transform @ object_grasp_positions[obj_index].flatten()
        position = position[:3]
        grasp_angle = object_grasp_angles[obj_index] + np.arctan(transform[1, 0] / transform[0, 0])

        # Visualize grasp position using a big red sphere
        markers.append(sim.SphereMarker(position, radius=0.02))

        # Attempt grasping
        grasp_success = env.execute_grasp(position, grasp_angle)
        print(f"Grasp success: {grasp_success}")

        if grasp_success:  # Move the object to another tote
            is_grasped[obj_index] = True

            # Get a list of robot configurations in small step sizes
            path_conf = main.rrt(env.robot_home_joint_config,
                                 env.robot_goal_joint_config, main.MAX_ITERS, main.delta_q, 0.5, env, distance=0.2)
            if path_conf is None:
                print("No collision-free path is found within the time budget. Continuing ...")
            else:
                # Execute the path while visualizing the location of joint 5
                env.set_joint_positions(env.robot_home_joint_config)
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.05)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim.SphereMarker(link_state[0], radius=0.02))

                print("Path executed. Dropping the object")

                # Drop the object
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()

                # Retrace the path to original location
                for joint_state in reversed(path_conf):
                    env.move_joints(joint_state, speed=0.1)

            p.removeAllUserDebugItems()

        markers = None
        env.robot_go_home()
