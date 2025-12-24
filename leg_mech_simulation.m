function leg_mech_simulation()
% Procedural Locomotion Simulation: 3-DOF Leg Terrain Adaptation.
%   Dependencies: Robotics System Toolbox
%   Author: Syaqina Octavia Rizha
%   Date: December 2025

    %% === CONFIGURATION & INITIALIZATION ===
    clc; clear; close all;
    
    % 1. Kinematic Parameters [Hip, Thigh, Shin]
    robot.L = [0.5, 0.8, 0.8]; 
    
    % 2. Initial State (Tucked in / Resting pose)
    % Initial position is close to the body
    robot.q = [0.0; 0.5; -1.5]; 
    
    % 3. Target Definition (LONG DISTANCE REACH)
    % We move the target far to X=1.4 and Y=0.6 to force leg extension.
    % This pushes the kinematic chain near its limit (Singularity Zone).
    target.pos = [1.4; 0.6; -0.4]; 
    
    % Orientation: Align foot with a steeper slope (Pitch = 30 deg)
    target.euler = [0, deg2rad(30), 0]; 
    target.quat = quaternion(target.euler, 'euler', 'ZYX', 'frame');

    % 4. Solver Hyperparameters
    solver.max_iter = 150;      % More iterations for longer move
    solver.tol_pos = 1e-4;      
    solver.tol_rot = 1e-4;      
    solver.lambda = 0.5;        % Higher damping for stability at full extension
    solver.step_size = 0.4;     % Slower step for smoother visualization

    %% === VISUALIZATION SETUP ===
    hFig = figure('Name', 'Long-Reach IK Simulation', 'Color', 'w');
    axes_handle = axes('Parent', hFig);
    grid on; axis equal; hold on;
    
    % Better view angle to see the extension
    view(45, 25); 
    
    % Draw "Ground"
    patch([-2 3 3 -2], [-2 -2 2 2], [-1.5 -1.5 -1.5 -1.5], ...
          [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    
    % Draw DISTANT TARGET (Red Dot)
    plot3(target.pos(1), target.pos(2), target.pos(3), ...
          'r.', 'MarkerSize', 35, 'DisplayName', 'Target Footprint');
      
    % Robot Plot Placeholder
    hRobot = plot3([0], [0], [0], '-o', 'LineWidth', 3, ...
                   'MarkerSize', 8, 'Color', [0.2 0.2 0.2], ...
                   'MarkerFaceColor', 'k');
    
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    xlim([-1 2.5]); ylim([-1 2]); zlim([-2 2]); % Expanded axis
    title('Initializing Long-Reach Solver...');
    
    %% === MAIN SOLVER LOOP ===
    
    for k = 1:solver.max_iter
        % 1. Forward Kinematics
        [curr_pos, curr_quat, joint_pts] = forward_kinematics(robot.q, robot.L);
        
        % 2. Error Calculation
        err_pos = target.pos - curr_pos;
        
        % Orientation Error
        q_diff = target.quat * conj(curr_quat);
        err_rot = rotvec(q_diff)';    
        error_vec = [err_pos; err_rot];
    
        % 3. Check Convergence
        if norm(err_pos) < solver.tol_pos && norm(err_rot) < solver.tol_rot
            fprintf('Target Reached at iter %d.\n', k);
            title(sprintf('Target Reached | Iter: %d', k));
            break;
        end
        
        % 4. Jacobian & DLS
        J = calculate_numeric_jacobian(robot.q, robot.L);
        
        % DLS Equation
        J_t = J';
        damped_inv = (J_t * J + solver.lambda^2 * eye(3)) \ J_t;
        dq = damped_inv * error_vec;
        
        % Update Joints
        robot.q = robot.q + (dq * solver.step_size);
        
        % 5. Update Visual (Slower pause for better visibility)
        set(hRobot, 'XData', joint_pts(1,:), ...
                    'YData', joint_pts(2,:), ...
                    'ZData', joint_pts(3,:));
        title(sprintf('Extending... Iter: %d | Err: %.3f', k, norm(err_pos)));
        
        pause(0.05); % Slow down animation
    end
end

%% === HELPER FUNCTIONS ===

function [pos, orient, points] = forward_kinematics(q, L)
    % q(1): Hip Yaw, q(2): Thigh Pitch, q(3): Knee Pitch
    q1 = quaternion([0, 0, q(1)], 'euler', 'XYZ', 'frame'); 
    q2 = quaternion([0, q(2), 0], 'euler', 'XYZ', 'frame'); 
    q3 = quaternion([0, q(3), 0], 'euler', 'XYZ', 'frame'); 
    
    p0 = [0; 0; 0]; 
    p1 = p0 + [0; 0; -0.2]; 
    
    acc_q2 = q1 * q2;
    v2 = rotatepoint(acc_q2, [L(1), 0, 0]);
    p2 = p1 + v2';
    
    acc_q3 = acc_q2 * q3;
    v3 = rotatepoint(acc_q3, [L(2), 0, 0]);
    p3 = p2 + v3';
    
    pos = p3;
    orient = acc_q3;
    points = [p0, p1, p2, p3];
end

function J = calculate_numeric_jacobian(q, L)
    delta = 1e-5;
    J = zeros(6, 3);
    [p_curr, q_curr, ~] = forward_kinematics(q, L);
    
    for i = 1:3
        q_new = q;
        q_new(i) = q_new(i) + delta;
        [p_new, q_new_quat, ~] = forward_kinematics(q_new, L);
        
        J(1:3, i) = (p_new - p_curr) / delta;
        q_diff = q_new_quat * conj(q_curr);
        J(4:6, i) = rotvec(q_diff)' / delta;
    end
end