%% N-step ahead predictor
clc; close all; clear all;



fs = 100;                    % Sampling frequency (samples per second)
dt = 1/fs;                   % seconds per sample
StopTime = 1;                % seconds
t = (0:dt:StopTime)';        % seconds
F = 1;                       % Sine wave frequency (hertz)


% use this for noiseless case
load('u_data.mat')
load('y_data.mat')


load('weight1.mat')
load('weight2.mat')
load('weight3.mat')

weights = struct('weight1',weight1,'weight2',weight2,'weight3',weight3);



%%
n_basis = length(weight1(:,1));
Tini = (length(weight1(1,:))+1)/2;
Q = 10*eye(n_basis); 
R=  1;

N = 10; %prediction horizon
Phi = []; Y = [];

y_ini = ones(Tini,1)*y_data(1)
u_ini = zeros(Tini-1,1)

%% recompute Theta 
for i = 1:length(y_data)-N;
if i == 1
y_ini = [y_ini(2:end);y_data(i)];
u_ini = u_ini;
end
if i >= 2 
y_ini = [y_ini(2:end);y_data(i)];
u_ini = [u_ini(2:end);u_data(i-1)];
end

uf = u_data(i:i+N-1)';

Phi = [Phi tanh_nn(weights,u_ini,uf,y_ini)];
out_vector = [y_data(i+1);y_data(i+2);y_data(i+3);y_data(i+4);y_data(i+5);y_data(i+6);y_data(i+7);y_data(i+8);y_data(i+9);y_data(i+10)];
% Y = [Y out_vector];
end


%% Learn the theta state model

Phi_f_i = Phi(1:n_basis,2:end)

Phi_f = []
for j = 1:length(Phi_f_i(1,:))-N+1
Phi_f_i_i = [];
for i = j:j+N-1
    Phi_f_i_i = [Phi_f_i_i;Phi_f_i(:,i)];
end 
Phi_f = [Phi_f,Phi_f_i_i];
end 

Phi_p = Phi(:,1:length(Phi_f));


%% end state definition
Theta = Phi_f*pinv(Phi_p)
Pm    = Theta(:,1:n_basis);
Gamma = Theta(:,n_basis+1:end);


%% State space matrices
A = Pm(1:n_basis,1:n_basis)
B = Gamma(1:n_basis,1)
Co = ctrb(A,B)
rank(Co)

%% Terminal set
 % clearvars -except A B n_basis

Y_var = sdpvar(1,n_basis)
O_var = sdpvar(n_basis,n_basis,'symmetric','real')

M_var = [O_var,         (A*O_var+B*Y_var).',        O_var,                      Y_var.';
    (A*O_var+B*Y_var),  O_var,                      zeros(n_basis,n_basis),     zeros(n_basis,1);
    O_var,              zeros(n_basis,n_basis),     inv(Q),                     zeros(n_basis,1);
    Y_var,              zeros(1,n_basis),           zeros(1,n_basis),           inv(R)]


objective = [ norm(O_var - 10*eye(n_basis)) ];
constraints = [M_var >= 0.000001*eye(1)];
options = sdpsettings('solver', 'mosek', 'verbose', 0, 'debug', 0)

optimize(constraints,objective,options)

Y_opt = value(Y_var)
O_opt = value(O_var)


M_test = [O_opt,        (A*O_opt+B*Y_opt).',        O_opt,                      Y_opt.';
    (A*O_opt+B*Y_opt),  O_opt,                      zeros(n_basis,n_basis),     zeros(n_basis,1);
    O_opt,              zeros(n_basis,n_basis),     inv(Q),                     zeros(n_basis,1);
    Y_opt,              zeros(1,n_basis),           zeros(1,n_basis),           inv(R)]



min(eig(M_test))
max(eig(M_test))

P = inv(O_opt)

eig(P)

K = Y_opt*P


Acl = A+B*K

test = abs(eig(Acl))
model   = LTISystem('A', Acl);

%% Compute the Terminal set

Ax = 0.1*[eye(n_basis);-eye(n_basis)]
bx = ones(2*n_basis,1)

Au = [2;-2]
bu = ones(2,1)

Xset = Polyhedron(Ax,bx)

Uset = Polyhedron(Au, bu);
XUset = Polyhedron(Au*K,bu) & Xset

InvSet = model.invariantSet('X',XUset);

max(abs(eig(A+B*K)))

MN = InvSet.A
bN = InvSet.b

save('invariant.mat',"Theta","A","B","K","bN","MN","P","Q","R")
