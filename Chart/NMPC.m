close all; clear all; clc;

M = 1   ;       % mass of the pendulum
k0 = 0.33
hd = 1.1
Ts = 1/30;

n = 2
m = 1

Q = [10,0;0,10] 
R =  1;
P = 1000*eye(n);


% Dimensions of the network
load('data/weight3.mat')
N = length(weight3(:,1));               % Prediction horizon
clear weight3



r = 0*ones(200,1)
k_sim = length(r)-N;

%% Setup the solver
u = sdpvar(N,1);
x = sdpvar(n,N);
x0 = sdpvar(n,1); 

objective = 0;
constraints = [];

x(1,1) = x0(1) + Ts*x0(2);
x(2,1) = x0(2) - k0*Ts/M*exp(-x0(1))*x0(1) - Ts*hd/M*x0(2) + Ts/M*u(1);

for k = 1:N-1
    objective = objective + x(:,k)'*Q*x(:,k) + u(k)'*R*u(k)
    constraints = [constraints,  -0.5<=u(k)<=0.5];
    x(1,k+1) = x(1,k) + Ts*x(2,k);
    x(2,k+1) = x(2,k) - k0*Ts/M*exp(-x(1,k))*x(1,k) - Ts*hd/M*x(2,k) + Ts/M*u(k+1);
end

objective = objective + x(:,N)'*P*x(:,N) + u(N)'*R*u(N);

Parameters = {x0};
Outputs = {u,x};

options = sdpsettings('solver', 'fmincon', 'verbose', 0, 'debug', 0);

controller = optimizer(constraints, objective, options, Parameters, Outputs);


%% initial conditions
xx1MPC(1) = -0.4;
xx2MPC(1) = 0;
yNMPC = xx1MPC;


for i = 1:k_sim;
i
t_vec_NMPC(i+1) = i*Ts;
tic;
X0 = [xx1MPC(i);xx2MPC(i)];

OUT = controller({X0});
uNMPC(i) = OUT{1}(1);

%%% output update MPC
xx1MPC(i+1) = xx1MPC(i) + Ts*xx2MPC(i);
xx2MPC(i+1) = xx2MPC(i) - k0*Ts/M*exp(-xx1MPC(i))*xx1MPC(i) - Ts*hd/M*xx2MPC(i) + Ts/M*uNMPC(i);    
yNMPC(i+1) = xx1MPC(i+1);

end


%% make eps figure
r = r(1:length(t_vec_NMPC)).';
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

save('data/NMPC','yNMPC','uNMPC','t_vec_NMPC')