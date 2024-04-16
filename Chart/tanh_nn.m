function [phi1_nn] = tanh_nn(param,uu,u,y)
weight1 = double(param.weight1);
weight2 = double(param.weight2);

x1 =  tanh(weight1*[uu;y]);  
x1 =  tanh(weight2*x1);  
phi1_nn = [x1;u.'];
end