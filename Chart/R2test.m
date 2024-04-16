close all; clear all; clc;

T_ini = 5;
n_basis = 10;


load('X_test.mat')
load('y_test.mat')
load('weight1.mat')
load('weight2.mat')
load('weight3.mat')

load('SPCTheta.mat')


X_test = double(X_test)'
y_test = double(y_test)'
weight1 = double(weight1)
weight2 = double(weight2)
weight3 = double(weight3)




for i = 1:length(X_test(1,:))
    %% test data 
    test_i = tanh(weight1*X_test(1:2*T_ini-1,i));
    test_i = tanh(weight2*test_i);
    y_KDPC_hat(:,i) = weight3*[test_i;X_test(T_ini*2:end,i)];

end 

y_SPC_hat = Theta_SPC*X_test;

%% Compute R2 value

y_bar = mean(y_test,2)


R2_Koop = 1-sum((y_KDPC_hat-y_test).^2,2)./sum((y_test-y_bar).^2,2)
R2_SPC = 1-sum((y_SPC_hat-y_test).^2,2)./sum((y_test-y_bar).^2,2)

