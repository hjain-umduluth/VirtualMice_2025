% Define a function to calculate tumor MLE
% The function takes in the following input arguments
% 1) time data (array)
% 2) tumor volume data (array)
% 3) tumor volume standard deviation (array)
% 4) initial tumor volume (array)
% 5) The model
% 6) initial parameters (array)
% 7) parameters lower bound (array)
function F = tumorMLE(time, tumorVolume, tumorSD, y0, model, paramsFit, paramsLb)

    % Utility function to pass into lsqcurvefit to find the MLE
    function fun = soln(pars, model, y0, time, tumorSD)
        % Set the error tolerances for ode solver
        options = odeset('RelTol',1e-5,'AbsTol',1e-7);
    
        % Solve the ODE & get fitted values
        [t,G] = ode23s(model, time, y0, options, pars);
        tmp = interp1(t, G, time, 'linear', 'extrap');

        % Apply weights
        fun = tmp ./ tumorSD;
    end

    % Get fitted parameters
    func = @(pars, tdata) soln(pars, model, y0, time, tumorSD);
    [parfit, resnorm] = lsqcurvefit(func, paramsFit, time, tumorVolume ./ tumorSD, paramsLb);
    F = parfit;
end