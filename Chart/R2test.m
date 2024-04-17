close all; clear all; clc;


load('data/u_data_test.mat')
load('data/y_data_test.mat')
load('data/weight1.mat')
load('data/weight2.mat')
load('data/weight3.mat')
load('data/SPCTheta.mat')

weight1 = double(weight1)
weight2 = double(weight2)
weight3 = double(weight3)


n_basis = length(weight1(:,1));
Tini = (length(weight1(1,:))+1)/2;
N = length(weight3(:,1));               % Prediction horizon



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Generate test data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_test = [];
y_test = [];
i = 1;
for i = 1:length(u_data_test)-N-Tini+2
    i
X_test = [X_test;[u_data_test(i:i+Tini-2)',y_data_test(i:i+Tini-1)',u_data_test(i+Tini-1:i+Tini+N-2)']];
y_test = [y_test,y_data_test(i:i+N-1)];
end

X_test = X_test'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(X_test(1,:))
    %% test data 
    test_i = tanh(weight1*X_test(1:2*Tini-1,i));
    test_i = tanh(weight2*test_i);
    y_KDPC_hat(:,i) = weight3*[test_i;X_test(Tini*2:end,i)];

end 

y_SPC_hat = Theta_SPC*X_test;

%% Compute R2 value

y_bar = mean(y_test,2);


R2_Koop = 1-sum((y_KDPC_hat-y_test).^2,2)./sum((y_test-y_bar).^2,2);
R2_SPC = 1-sum((y_SPC_hat-y_test).^2,2)./sum((y_test-y_bar).^2,2);

