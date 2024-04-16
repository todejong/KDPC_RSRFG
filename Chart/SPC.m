%% N-step ahead predictor
clc; close all; clear all;

M = 1   ;       % mass of the pendulum
k0 = 0.33
hd = 1.1
Ts = 1/30;

r = 0*ones(200,1)
Tini = 5

Q = 1; 
R=  0.001;
P = 1000;

%%
N = 10; %prediction horizon
k_sim = length(r)-N;

%% recompute Theta 
load('X_train.mat')
load('y_train.mat')

X_train = double(X_train).'
y_train = double(y_train).'


%% Learn the theta state model
Theta_SPC = y_train*pinv(X_train)
save('SPCTheta','Theta_SPC')

%% end state definition
P1 = Theta_SPC(:,1:Tini-1)
P2 = Theta_SPC(:,Tini:2*Tini-1)
Gamma = Theta_SPC(:,2*Tini:end)


%% Simulate the system
Psi = eye(N)*R
Omega = eye(N)*Q
Omega(end,end) = P;


uSPC = sdpvar(N,1);
ySPC = sdpvar(N,1);
yini = sdpvar(Tini,1); 
uini = sdpvar(Tini-1,1);

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
xx2(1) = -0.2;
uSPC = [];
th=[];
NL_part_all = [];
Z0 = [];
t_SPC = 0

Y_ini = ones(Tini,1)*ySPC(1)
U_ini = zeros(Tini-1,1)

%%
for i = 1:k_sim;
i
tic;
t_SPC(i+1) = i*Ts;

%%% u_ini & y_ini   
if i == 1
Y_ini = [Y_ini(2:end);ySPC(i)];
U_ini = U_ini;
end
if i >= 2 
Y_ini = [Y_ini(2:end);ySPC(i)];
U_ini = [U_ini(2:end);uSPC(i-1)];
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
subplot(2,1,2)
hold on;
plot(t_SPC(1:end-1),uSPC,'LineWidth',3,'Color',"#0072BD");
yline(3,'LineWidth',1,'LineStyle','--','Color','red');
yline(-3,'LineWidth',1,'LineStyle','--','Color','red');
ylabel('$u$',Interpreter='latex')
legend('KDPC','LQR','MPC','Location','southeast');
axis tight 
grid on;


save('SPC','t_SPC','ySPC','uSPC')
