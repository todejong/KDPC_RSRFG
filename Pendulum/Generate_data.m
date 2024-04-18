close all; clear all; clc;

%% N-step ahead predictor

M = 1.   ;       % mass of the pendulum
L = 1.   ;       % lenght of the pendulum
b = 0.1  ;       % friction coefficient
g = 9.81  ;      % acceleration of gravity
J = 1/3*M*L^2 ;  % moment of inertia
Ts = 1/30;

Range = [-4,4]
SineData = [25, 40, 1]
Band = [0, 1]
NumPeriod = 1
Period = 1500
Nu = 1
TrainTest = 2/3

u_data = idinput([Period 1 NumPeriod],'sine',Band,Range,SineData)';
% u_data = idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)';
% u_data = zeros(1,length(u_data))
t = 0:Ts:(length(u_data)-1)*Ts


x1 = 0
x2 = 0

for i = 1:length(u_data)
    t(i) = i*Ts;
    x1(i+1) = (1-b*Ts/J)*x1(i) + (Ts)/J*u_data(i) - (M*L*g*Ts)/(2*J)*sin(x2(i));
    x2(i+1) = Ts*x1(i) + x2(i);

    y_data(i) = x2(i);
end 


u_data_test = u_data(TrainTest*end+1:end);
y_data_test = y_data(TrainTest*end+1:end);
t_test = t(TrainTest*end+1:end);
u_data = u_data(1:TrainTest*end);
y_data = y_data(1:TrainTest*end);
t_train = t(1:TrainTest*end);


curr_fig = figure;
curr_axes1=axes('Parent',curr_fig,'FontSize',11,'FontName','Times New Roman');
box(curr_axes1,'on');
hold(curr_axes1,'all');
%your plots
subplot(2,1,1)
hold on
plot(t_train,u_data,'LineWidth',1)
plot(t_test,u_data_test,'LineWidth',1)
ylabel('$u(k)$',Interpreter='latex')
legend('train','test',Interpreter='latex')
axis tight
subplot(2,1,2)
hold on
plot(t_train,y_data(1,:),'LineWidth',1)
plot(t_test,y_data_test(1,:),'LineWidth',1)
ylabel('$y(k)$',Interpreter='latex')
xlabel('$t$[s]',Interpreter='latex')
axis tight
%your plots
set(gca,'TickLabelInterpreter','Latex')
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
% you can change 9 and 6.3 to change the ratios of the plot...
savefig('figures/identification_pend.fig') %change it with the name you want to give to the .fig plot
print -depsc figures/identification_pend %change it with the name you want to give to the .eps figure


u_data = u_data.'
y_data = y_data.'
save('data/u_data',"u_data")
save('data/y_data',"y_data")


u_data_test = u_data_test.'
y_data_test = y_data_test.'
save('data/u_data_test',"u_data_test")
save('data/y_data_test',"y_data_test")