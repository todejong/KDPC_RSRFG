%% N-step ahead predictor
clc; close all; clear all;

M = 1   ;       % mass of the pendulum
k0 = 0.33
hd = 1.1
Ts = 1/30;

fs = 100;                    % Sampling frequency (samples per second)
dt = 1/fs;                   % seconds per sample
StopTime = 1.5;                % seconds
t = (0:dt:StopTime)';        % seconds
F = 1;                       % Sine wave frequency (hertz)
r = 0.5 + sin(2*pi*F*t);           % Reference
r = 0*ones(length(r),1)
T_ini = 5

Q = 10; 
R=  1;
P = 1000;
lambda = 1e+7;

% use this for noiseless case

%%
N = 10; %prediction horizon
k_sim = length(r)-N;
Phi = []; Y = [];

%% recompute Theta 
% for i = 1:length(u_data)-T_ini-N
% U(:,i) = u_data(i:T_ini+N+i-1);
% Y(:,i) = y_data(i+1:T_ini+i+N);
% end
% Up = U(1:T_ini,:)
% Uf = U(T_ini+1:end ,:)
% Yp = Y(1:T_ini,:)
% Yf = Y(T_ini+1:end ,:)

load('X.mat')
load('y.mat')
load('X_train.mat')
load('y_train.mat')

X_train = double(X_train).'
y_train = double(y_train).'


%% Learn the theta state model
Theta_SPC = y_train*pinv(X_train)
save('SPCTheta','Theta_SPC')

%% end state definition
P1 = Theta_SPC(:,1:T_ini-1)
P2 = Theta_SPC(:,T_ini:2*T_ini-1)
Gamma = Theta_SPC(:,2*T_ini:end)


%% Simulate the system
Psi = eye(N)*R
Omega = eye(N)*Q
Omega(end,end) = P;


uSPC = sdpvar(N,1);
ySPC = sdpvar(N,1);
yini = sdpvar(T_ini,1); 
uini = sdpvar(T_ini-1,1);

objective = ySPC'*Omega*ySPC+(uSPC)'*Psi*(uSPC);  %  + lambda 

constraints = [ySPC == P1*uini + P2*yini + Gamma*uSPC];

for k = 1:N
    constraints = [constraints,  -0.5<=uSPC(k)<=0.5];
end
Parameters = {uini,yini};
Outputs = {uSPC,ySPC};

options = sdpsettings('solver', ['QUADPROG' ...
    ''], 'verbose', 0, 'debug', 0);

%options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0, 'osqp.eps_abs', 1e-3, 'osqp.eps_rel', 1e-3);
controller = optimizer(constraints, objective, options, Parameters, Outputs);

%% initial conditions
ySPC(1) = -0.3;
xx1(1) = ySPC(1);
omega = -0.2;
xx2(1) = omega;
u_mpc = 0;
uSPC = [];
th=[];
NL_part_all = [];
Z0 = [];
t_SPC = 0

%%
for i = 1:k_sim;
i
tic;
t_SPC(i+1) = i*dt;

%%% u_ini & y_ini   
if i == 1
Y_ini = [ySPC(1);ySPC(1);ySPC(1);ySPC(1); ySPC(i)];
U_ini = [0;0;0;0];
end
if i == 2 
Y_ini = [ySPC(1);ySPC(1);ySPC(1);ySPC(i-1); ySPC(i)];
U_ini = [0;0;0;uSPC(i-1)];
end
if i == 3 
Y_ini = [ySPC(1);ySPC(1);ySPC(i-2);ySPC(i-1); ySPC(i)];
U_ini = [0;0;uSPC(i-2);uSPC(i-1)];
end
if i == 4 
Y_ini = [ySPC(1);ySPC(i-3);ySPC(i-2);ySPC(i-1); ySPC(i)];
U_ini = [0;uSPC(i-3);uSPC(i-2);uSPC(i-1)];
end
if i >= 5 
Y_ini = [ySPC(i-4);ySPC(i-3);ySPC(i-2);ySPC(i-1); ySPC(i)];
U_ini = [uSPC(i-4);uSPC(i-3);uSPC(i-2);uSPC(i-1)];
end


OUT = controller({U_ini,Y_ini});

Uk = OUT{1};

u_mpc = Uk(1);
uSPC = [uSPC u_mpc];
th=[th;toc];

%%% output update
xx1(i+1) = xx1(i) + Ts*xx2(i);
xx2(i+1) = xx2(i) - k0*Ts/M*exp(-xx1(i))*xx1(i) - Ts*hd/M*xx2(i) + Ts/M*u_mpc;    
ySPC(i+1) = xx1(i+1);
end

%% New plot

r = r(1:length(t_SPC)).'
curr_fig = figure;
curr_axes1=axes('Parent',curr_fig,'FontSize',11,'FontName','Times New Roman');
box(curr_axes1,'on');
hold(curr_axes1,'all');
%your plots
subplot(2,1,1)
hold on;
plot(t_SPC,r,'LineWidth',3,'Color',"#7E2F8E");
plot(t_SPC,ySPC,'LineWidth',3,'Color',"#0072BD");
legend('','KDPC','LQR','MPC','Location','northeast');
ylabel('$x_2$',Interpreter='latex')
axis tight 
grid on
xlim([0,1.4])
subplot(2,1,2)
hold on;
plot(t_SPC(1:end-1),uSPC,'LineWidth',3,'Color',"#0072BD");
yline(3,'LineWidth',1,'LineStyle','--','Color','red');
yline(-3,'LineWidth',1,'LineStyle','--','Color','red');
ylabel('$u$',Interpreter='latex')
legend('KDPC','LQR','MPC','Location','southeast');
axis tight 
grid on;
xlim([0,1.4])


save('SPC','t_SPC','ySPC','uSPC')
