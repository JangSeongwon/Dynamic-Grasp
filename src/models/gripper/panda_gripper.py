<mujoco model="panda_hand">
    <size njmax="500" nconmax="100" />
    <asset>
        <mesh name="hand" file="meshes/panda_gripper/hand.stl" />
        <mesh name="hand_vis" file="meshes/panda_gripper/hand_vis.stl" />
        <mesh name="finger" file="meshes/panda_gripper/finger.stl" />
        <mesh name="finger_vis" file="meshes/panda_gripper/finger_vis.stl" />
        <mesh name="finger_vis2" file="meshes/panda_gripper/finger_longer.stl" />
        <mesh name="camera1st" file="meshes/panda_gripper/camera.STL" />
    </asset>

    <actuator>
        <position ctrllimited="true" ctrlrange="0.00 0.04" joint="finger_joint1" kp="10000" name="gripper_joint1" forcelimited="true" forcerange="-20 20"/>
        <position ctrllimited="true" ctrlrange="-0.04 -0.00" joint="finger_joint2" kp="10000" name="gripper_joint2" forcelimited="true" forcerange="-20 20"/>
    </actuator>

    <worldbody>
      <body name="gripper" pos="0 0 0">

            <site name="ft_frame" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" group="1"></site>

            <inertial pos="0 0 0.17" quat="0.707107 0.707107 0 0" mass="0.0" diaginertia="0.09 0.07 0.05" />
            <geom pos="0 0 0." quat="0.707107 0 0 0.707" type="mesh" contype="0" conaffinity="0" group="1" mesh="hand_vis" name="hand_visual" rgba="1 1 1 1"/>
            <geom pos="0 0 0." quat="0.707107 0 0 0.707" type="mesh" mesh="hand"  group="0" name="hand_collision"/>


            <site name="grip_site" pos="0 0 0.1" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" group="1"/>

            <!-- First person Camera -->
            <camera mode="fixed" name="1stcamera" pos="0 -0.06 0.05" euler="3.14159265358979 0 0" fovy="25"/>

            <body name="leftfinger" pos="0 0 0.0524" quat="0.707107 0 0 0.70">
                <inertial pos="0 0 0.05" mass="0.1" diaginertia="0.01 0.01 0.005" />
                <joint name="finger_joint1" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="0.00 0.04" damping="100"/>
                <geom quat="0.707107 0.707107 0 0" type="mesh" contype="0" conaffinity="0" group="1" mesh="finger_vis" name="finger1_visual"/>
                <geom type="mesh" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4" mesh="finger" name="finger1_collision"/>
                <body name="finger_joint1_tip" pos="0 0.0085 0.059">
                    <inertial pos="0 0 0" quat="0 0 0 1" mass="0.01" diaginertia="0.01 0.01 0.01" />
                    <geom size="0.008 0.004 0.008" pos="0 -0.005 -0.015" quat="0 0 0 1" type="box"  conaffinity="1" contype="1" name="finger1_tip_collision"/>
                </body>
            </body>
            <body name="rightfinger" pos="0 0 0.0524" quat="0.707107 0 0 0.70">
                <inertial pos="0 0 0.05" mass="0.1" diaginertia="0.01 0.01 0.005" />
                <joint name="finger_joint2" pos="0 0 0" axis="0 1 0" type="slide" limited="true" range="-0.04 -0.00" damping="100"/>
                <geom quat="0 0 0.707107 0.707107" type="mesh" contype="0" conaffinity="0" group="1" mesh="finger_vis" name="finger2_visual"/>
                <geom quat="0 0 0 1" type="mesh" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4" mesh="finger" name="finger2_collision"/>
                <body name="finger_joint2_tip" pos="0 -0.0085 0.059">
                    <inertial pos="0 0 0" quat="0 0 0 1" mass="0.01" diaginertia="0.01 0.01 0.01" />
                    <geom size="0.008 0.004 0.008" pos="0 0.005 -0.015" quat="0 0 0 1" type="box"  conaffinity="1" contype="1" name="finger2_tip_collision"/>
                </body>
            </body>
      </body>
    </worldbody>
      <sensor>
        <force name="force_ee" site="ft_frame"/>
        <torque name="torque_ee" site="ft_frame"/>
      </sensor>
</mujoco>
