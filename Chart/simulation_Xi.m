%% N-step ahead predictor
clc; close all; clear all;



M = 1   ;       % mass of the pendulum
k0 = 0.33
hd = 1.1
Ts = 1/30;

n_basis = 8;

fs = 100;                    % Sampling frequency (samples per second)
dt = 1/fs;                   % seconds per sample
StopTime = 2;                % seconds
t = (0:dt:StopTime)';        % seconds
F = 1;                       % Sine wave frequency (hertz)
r = 0.5 + sin(2*pi*F*t);     % Reference
r = 0*ones(length(r),1)


Q = 10*eye(n_basis); 
R=  1;
lambda = 1e+9;

% use this for noiseless case
load('u_data.mat')
load('y_data.mat')


load('weight1.mat')
load('weight2.mat')
load('weight3.mat')

weights = struct('weight1',weight1,'weight2',weight2,'weight3',weight3);


%%
N = 10; %prediction horizon
k_sim = length(r)-N;
Phi = []; Y = [];
load('invariant.mat')



%% Simulate the system

Psi = kron(eye(N), R);
Omega = kron(eye(N), Q)


Omega = blkdiag(Omega(1:end-n_basis,1:end-n_basis),P)



u = sdpvar(N,1);
Z = sdpvar(N*n_basis,1);
%u_ini = sdpvar(Tini-1, 1);
%y_ini = sdpvar(Tini, 1);
nl_part = sdpvar(length(weight2),1);
z_part = sdpvar(length(weight2),1);
z0 = sdpvar(length(weight2),1); 
xi = sdpvar(1,1);

objective = Z'*Omega*Z+(u)'*Psi*(u) + z0'*Q*z0 + lambda*(z_part-nl_part)'*(z_part-nl_part)*xi^2;  %  + lambda 

constraints = [z0 == (1-xi)*nl_part + xi*z_part];


constraints = [constraints, 0<=xi<=1];
constraints = [constraints, MN*Z((end-n_basis+1):end,1)<=bN];
constraints = [constraints, Z == Theta*[z0;u]];
for k = 1:N
    constraints = [constraints,  -0.5<=u(k)<=0.5];
end
Parameters = {nl_part,z_part};
Outputs = {u,Z,xi,z0};

options = sdpsettings('solver', ['QUADPROG' ...
    ''], 'verbose', 0, 'debug', 0);

%options = sdpsettings('solver', 'quadprog', 'verbose', 0, 'debug', 0, 'osqp.eps_abs', 1e-3, 'osqp.eps_rel', 1e-3);
controller = optimizer(constraints, objective, options, Parameters, Outputs);

%% initial conditions
u = [];
th=[];
NL_part_all = [];
Z0 = [];
t_vec = 0
%%
y(1) = -0.3;
xx2(1) = -0.2;
xx1(1) = y(1);
u_mpc = 0;


%% simulation 
for i = 1:k_sim;
i
t_vec(i+1) = i*dt;
tic;
%%% u_ini & y_ini   
if i == 1
y_ini = [y(1);y(1);y(1);y(1); y(i)];
u_ini = [0;0;0;0];
end
if i == 2 
y_ini = [y(1);y(1);y(1);y(i-1); y(i)];
u_ini = [0;0;0;u(i-1)];
end
if i == 3 
y_ini = [y(1);y(1);y(i-2);y(i-1); y(i)];
u_ini = [0;0;u(i-2);u(i-1)];
end
if i == 4 
y_ini = [y(1);y(i-3);y(i-2);y(i-1); y(i)];
u_ini = [0;u(i-3);u(i-2);u(i-1)];
end
if i >= 5 
y_ini = [y(i-4);y(i-3);y(i-2);y(i-1); y(i)];
u_ini = [u(i-4);u(i-3);u(i-2);u(i-1)];
end

clear Nl_part;
Nl_part = [];
out =  tanh(weight1*[u_ini;y_ini]);  
Nl_part =  tanh(weight2*out);  

NL_part_all = [NL_part_all, Nl_part];

if i < 5 
Z_part = Nl_part;
else 
Z_part = Zkm1;
end 
%%% mpc sol
Nl_part = double(Nl_part);
Z_part = double(Z_part);

OUT = controller({Nl_part,Z_part});

Uk = OUT{1};
Zkm1 = OUT{2}(1:n_basis);
Xi(i) = OUT{3};
Z0 = [Z0,OUT{4}];

u_mpc = Uk(1);
u = [u u_mpc];
th=[th;toc];

%%% output update    
xx1(i+1) = xx1(i) + Ts*xx2(i);
xx2(i+1) = xx2(i) - k0*Ts/M*exp(-xx1(i))*xx1(i) - Ts*hd/M*xx2(i) + Ts/M*u(i);    
y(i+1) = xx1(i+1);

end


e = y-r(1:length(y));



%% New plot

load('SPC.mat')
load('NMPC.mat')


r  = r(1:length(t_vec)).'


curr_fig = figure;
curr_axes1=axes('Parent',curr_fig,'FontSize',11,'FontName','Times New Roman');
box(curr_axes1,'on');
hold(curr_axes1,'all');
%your plots
subplot(2,1,1)
hold on;
plot(t_vec,r,'k','LineWidth',1.5);
plot(t_vec,y,'LineWidth',3,'Color',"#0072BD");
plot(t_SPC,ySPC,'LineWidth',3,'Color',"#EDB120");
plot(t_vec_NMPC,yNMPC,'LineWidth',3,'Color',"#D95319");
legend('','KDPC','SPC','NMPC','Location','southeast');
ylabel('$x_1$',Interpreter='latex')
axis tight 
grid on
xlim([0,1.6])
subplot(2,1,2)
hold on;
plot(t_vec(1:end-1),u,'LineWidth',3,'Color',"#0072BD");
plot(t_SPC(1:end-1),uSPC,'LineWidth',3,'Color',"#EDB120");
plot(t_vec_NMPC(1:end-1),uNMPC,'LineWidth',3,'Color',"#D95319");
yline(0.5,'r--','LineWidth',0.5)
yline(-0.5,'r--','LineWidth',0.5)
legend('KDPC','SPC','NMPC','','','Location','southeast');
ylabel('$u$',Interpreter='latex')
xlabel('$t$[s]',Interpreter='latex')
axis tight 
grid on;
xlim([0,1.6])
ylim([-0.6,0.6])
%your plots
set(gca,'TickLabelInterpreter','Latex')
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
% you can change 9 and 6.3 to change the ratios of the plot...
savefig('2chart.fig') %change it with the name you want to give to the .fig plot
print -depsc 2chart %change it with the name you want to give to the .eps figure



curr_fig = figure;
curr_axes1=axes('Parent',curr_fig,'FontSize',11,'FontName','Times New Roman');
box(curr_axes1,'on');
hold(curr_axes1,'all');
%your plots
subplot(2,1,1)
plot(t_vec(1:end-1),Xi,'LineWidth',3,'Color',"#0072BD")
ylabel('$\xi$',Interpreter='latex')
grid on;
xlim([0,1.6])
subplot(2,1,2)
hold on
for i = 1:length(NL_part_all(:,1))
plot(t_vec(1:end-1),NL_part_all(i,:),'LineWidth',3)
end
ylabel('$\varphi_i$',Interpreter='latex')
xlabel('$t [s]$',Interpreter='latex')
axis tight 
grid on;
xlim([0,1.6])
%your plots
set(gca,'TickLabelInterpreter','Latex')
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
% you can change 9 and 6.3 to change the ratios of the plot...
savefig('2chart2.fig') %change it with the name you want to give to the .fig plot
print -depsc 2chart2 %change it with the name you want to give to the .eps figure



