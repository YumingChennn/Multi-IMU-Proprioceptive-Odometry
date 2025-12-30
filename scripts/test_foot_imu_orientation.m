% Test different R_fm_list to find the correct foot IMU orientation
% This script compares transformed foot IMU data with body IMU data

clear; close all;

% Load data first
run('../kinematics_init_lc');
run('../param_init');
addpath("../rot_lib");
addpath("../common_func/");

% You need to set your bag file path here
file_path = '/home/ray/Multi-IMU-Proprioceptive-Odometry/dataset/output_ros1_5.bag';
param.fl_imu_topic = '/imu/data/fl';
param.fr_imu_topic = '/imu/data/fr';
param.rl_imu_topic = '/imu/data/rl';
param.rr_imu_topic = '/imu/data/rr';
param.body_imu_topic = '/imu/data/base_link';
param.mocap_topic = '/mocap/body/pose';
param.mocap_FR_topic = '/mocap/fr/pose';
param.joint_foot_topic = '/joint_states';

[sensor_data, param] = get_five_imu_sensor_data_from_rosbag(file_path, param);

% Test different R_fm_list configurations
disp('========================================');
disp('   Testing Foot IMU Orientations       ');
disp('========================================');

% Different rotation matrices to test
R_candidates = {
    eye(3),                    % Identity (no rotation)
    [-1  0  0; 0 -1  0; 0  0  1],  % 180 deg around Z
    [-1  0  0; 0  0 -1; 0 -1  0],  % A1 FL orientation (current)
    [ 0  1  0; 1  0  0; 0  0 -1],  % 90 deg rotations
    [ 0 -1  0; 1  0  0; 0  0  1],
    [ 1  0  0; 0 -1  0; 0  0 -1],
};

R_names = {
    'Identity (no rotation)',
    '180° around Z',
    'A1 FL orientation (current)',
    '90° variant 1',
    '90° variant 2', 
    'Mirror XY'
};

% Pick a moment when robot is standing still
still_idx = 150;
joint_angs = sensor_data.joint_ang.Data(still_idx,:)';
body_accel = sensor_data.accel_body_IMU.Data(still_idx,:)';

% Expected: body frame accelerometer should read ~[0, 0, -9.8] when still
disp(' ');
disp('Body IMU (standing still):');
fprintf('  Accel: [%.3f, %.3f, %.3f] m/s^2\n', body_accel);
fprintf('  Magnitude: %.3f m/s^2 (should be ~9.8)\n', norm(body_accel));

% Test FL leg (leg_id = 1)
leg_id = 1;
leg_joint_angs = joint_angs((leg_id-1)*3+1:(leg_id-1)*3+3);
R_bf = autoFunc_fk_pf_rot(leg_joint_angs, param.rho_opt_true(:,leg_id), param.rho_fix(:,leg_id));
foot_accel_raw = sensor_data.accel_fl_IMU.Data(still_idx,:)';

disp(' ');
disp('FL Foot IMU (raw data):');
fprintf('  Accel: [%.3f, %.3f, %.3f] m/s^2\n', foot_accel_raw);
fprintf('  Magnitude: %.3f m/s^2\n', norm(foot_accel_raw));

disp(' ');
disp('Testing different R_fm orientations for FL leg:');
disp('(Looking for the one that matches body IMU best)');
disp(' ');

best_error = inf;
best_idx = 1;

for i = 1:length(R_candidates)
    R_fm = R_candidates{i};
    
    % Transform foot IMU to body frame
    accel_body_frame = R_bf * R_fm * foot_accel_raw;
    
    % Compare with body IMU
    error = norm(accel_body_frame - body_accel);
    
    fprintf('%d. %s\n', i, R_names{i});
    fprintf('   Transformed: [%.3f, %.3f, %.3f] m/s^2\n', accel_body_frame);
    fprintf('   Error from body IMU: %.3f m/s^2\n', error);
    
    if error < best_error
        best_error = error;
        best_idx = i;
    end
    
    disp(' ');
end

disp('========================================');
fprintf('BEST MATCH: #%d - %s\n', best_idx, R_names{best_idx});
fprintf('Error: %.3f m/s^2\n', best_error);
disp('========================================');

% Generate code for mipo_conf_init.m
disp(' ');
disp('Suggested R_fm_list for mipo_conf_init.m:');
disp('(Assuming all 4 legs have similar orientation)');
disp(' ');
R_best = R_candidates{best_idx};
fprintf('param.R_fm_list = {[%2d %2d %2d;\n', R_best(1,:));
fprintf('              %2d %2d %2d;\n', R_best(2,:));
fprintf('              %2d %2d %2d],\n', R_best(3,:));
fprintf('            [%2d %2d %2d;\n', R_best(1,:));
fprintf('              %2d %2d %2d;\n', R_best(2,:));
fprintf('              %2d %2d %2d],\n', R_best(3,:));
fprintf('            [%2d %2d %2d;\n', R_best(1,:));
fprintf('              %2d %2d %2d;\n', R_best(2,:));
fprintf('              %2d %2d %2d],\n', R_best(3,:));
fprintf('            [%2d %2d %2d;\n', R_best(1,:));
fprintf('              %2d %2d %2d;\n', R_best(2,:));
fprintf('              %2d %2d %2d]};\n', R_best(3,:));

% Plot comparison
figure('Name', 'Foot IMU Orientation Test');
subplot(2,1,1);
hold on;
for i = 1:length(R_candidates)
    R_fm = R_candidates{i};
    accel_body_frame = R_bf * R_fm * foot_accel_raw;
    plot3(accel_body_frame(1), accel_body_frame(2), accel_body_frame(3), 'o', 'MarkerSize', 10);
end
plot3(body_accel(1), body_accel(2), body_accel(3), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
grid on; xlabel('X'); ylabel('Y'); zlabel('Z');
title('Foot IMU in Body Frame (different orientations)');
legend([R_names, 'Body IMU (target)'], 'Location', 'best');

subplot(2,1,2);
errors = zeros(length(R_candidates), 1);
for i = 1:length(R_candidates)
    R_fm = R_candidates{i};
    accel_body_frame = R_bf * R_fm * foot_accel_raw;
    errors(i) = norm(accel_body_frame - body_accel);
end
bar(errors);
ylabel('Error (m/s^2)');
xlabel('Orientation #');
title('Error from Body IMU');
set(gca, 'XTickLabel', 1:length(R_candidates));
grid on;
