close all; clear all; clc;

%% N-step ahead predictor
M = 1;       % mass of the pendulum
k0 = 0.33
hd = 1.1
Ts = 1/30;


Range = [-4, 4]
SineData = [25, 40, 1]
Band = [0, 1]
NumPeriod = 1
Period = 2000
Nu = 1
TrainTest = 2/3

u_data = idinput([Period 1 NumPeriod],'sine',Band,Range,SineData)';
% u_data = [u_data, idinput([Period 1 NumPeriod],'prbs',Band,Range,SineData)']
% u_data = zeros(1,length(u_data))

% u_data = block+noise
t = 0:Ts:(length(u_data)-1)*Ts


x1 = 0
x2 = 0


for k = 1:length(u_data)
    x1(k+1) = x1(k) + Ts*x2(k);
    x2(k+1) = x2(k) - k0*Ts/M*exp(-x1(k))*x1(k) - Ts*hd/M*x2(k) + Ts/M*u_data(k); 
    
    y_data(k) = x1(k);

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
xlim([t(1), t(end)])
subplot(2,1,2)
hold on
plot(t_train,y_data(1,:),'LineWidth',1)
plot(t_test,y_data_test(1,:),'LineWidth',1)
ylabel('$y(k)$',Interpreter='latex')
xlabel('$t$[s]',Interpreter='latex')
xlim([t(1), t(end)])
%your plots
set(gca,'TickLabelInterpreter','Latex')
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
% you can change 9 and 6.3 to change the ratios of the plot...
savefig('figures/identification.fig') %change it with the name you want to give to the .fig plot
print -depsc figures/identification %change it with the name you want to give to the .eps figure


u_data = u_data.'
y_data = y_data.'
save('data/u_data',"u_data")
save('data/y_data',"y_data")


u_data_test = u_data_test.'
y_data_test = y_data_test.'
save('data/u_data_test',"u_data_test")
save('data/y_data_test',"y_data_test")