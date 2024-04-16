close all; clear all; clc;

M = 1.   ;       % mass of the pendulum
L = 1.   ;       % lenght of the pendulum
b = 0.1  ;       % friction coefficient
g = 9.81  ;      % acceleration of gravity
J = 1/3*M*L^2 ;  % moment of inertia
Ts = 1/30;

n = 2
m = 1

Q = [10,0;0,10] 
R =  1;
P = 1000*eye(n);
N = 10; %prediction horizon

fs = 100;                    % Sampling frequency (samples per second)
dt = 1/fs;                   % seconds per sample
StopTime = 2;                % seconds
t = (0:dt:StopTime)';        % seconds
F = 1;                       % Sine wave frequency (hertz)
r = 0.5 + sin(2*pi*F*t);           % Reference
r = 0*ones(length(r),1)
k_sim = length(r)-N;

%% construct Gamma and Psi

%% Setup the solver
u = sdpvar(N,1);
x = sdpvar(n,N+1);
x0 = sdpvar(n,1); 

objective = 0
constraints = [];
constraints = [constraints, x(1,1) == (1-b*Ts/J)*x0(1) + (Ts)/J*u(1) - (M*L*g*Ts)/(2*J)*sin(x0(2))];
constraints = [constraints, x(2,1) == Ts*x0(1) + x0(2)];

for k = 1:N-1
    objective = objective + x(:,k)'*Q*x(:,k) + u(k)'*R*u(k)
    constraints = [constraints,  -3<=u(k)<=3];
    constraints = [constraints, x(1,k+1) == (1-b*Ts/J)*x(1,k) + (Ts)/J*u(k+1) - (M*L*g*Ts)/(2*J)*sin(x(2,k))];
    constraints = [constraints, x(2,k+1) == Ts*x(1,k) + x(2,k)];

end

objective = objective + x(:,N)'*P*x(:,N) 

Parameters = {x0};
Outputs = {u,x};

options = sdpsettings('solver', 'fmincon', 'verbose', 0, 'debug', 0);

controller = optimizer(constraints, objective, options, Parameters, Outputs);


%% initial conditions
xx1MPC(1) = 7;
xx2MPC(1) = 0.2;
yNMPC = xx2MPC;


for i = 1:k_sim;
i
t_vec_NMPC(i+1) = i*dt;
tic;
X0 = [xx1MPC(i);xx2MPC(i)];

OUT = controller({X0});
uNMPC(i) = OUT{1}(1);

%%% output update MPC
xx1MPC(i+1) = (1-b*Ts/J)*xx1MPC(i) + (Ts)/J*uNMPC(i) - (M*L*g*Ts)/(2*J)*sin(xx2MPC(i));
xx2MPC(i+1) = Ts*xx1MPC(i) + xx2MPC(i);
yNMPC(i+1) = xx2MPC(i+1);

end


%% make eps figure
r = r(1:length(t_vec_NMPC)).'
curr_fig = figure;
curr_axes1=axes('Parent',curr_fig,'FontSize',11,'FontName','Times New Roman');
box(curr_axes1,'on');
hold(curr_axes1,'all');
%your plots
subplot(2,1,1)
hold on;
plot(t_vec_NMPC,r,'LineWidth',3);
plot(t_vec_NMPC,yNMPC,'LineWidth',3);
legend('reference','KMPC','yLQR','Location','southeast');
ylabel('$x_1$',Interpreter='latex')
axis tight 
grid on
subplot(2,1,2)
hold on;
plot(t_vec_NMPC(1:end-1),uNMPC,'LineWidth',3);
yline(0.5,'LineWidth',1,'LineStyle','--','Color','red');
yline(-0.5,'LineWidth',1,'LineStyle','--','Color','red');
ylabel('$u$',Interpreter='latex')
xlabel('$k$',Interpreter='latex')
legend('KMPC','LQR','Location','southeast');
axis tight 
grid on;

save('NMPC','yNMPC','uNMPC','t_vec_NMPC')