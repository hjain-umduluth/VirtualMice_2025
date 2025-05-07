function F = relDE_logistic(n,r,pars)

rho = pars(1);
gamma = pars(2);

F = [rho*r(1)*(r(1)/gamma-1)];