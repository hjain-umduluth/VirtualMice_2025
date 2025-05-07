function SurvivalCurves_crjg

% INPUT THE DATA

%%%%%%%%%%%%%%
%%%  Data  %%%
%%%%%%%%%%%%%%

%%% Control
Time     = [1         6         8         11        13];      % days
TumVol   = [0.0940    0.4900    0.5700    1.0980    1.4460];      % tumor volume in cm^3
TumSD    = [0.0880    0.3200    0.3760    0.1360    0.3520];      % tumor SD 

%survTime     = [1         6         8         11        13       14      15];      % days
%survVol      = [0.0940    0.4900    0.5700    1.0980    1.4460   1.6487  1.8658];      % tumor volume in cm^3
%survProb     = [1         1         1         0.6       0.4      0.2     0];      % survival probability at day t in control

survTime     = [1         6.5       12        13.5     14.5    15];      % days
survVol      = [0.0940    0.4753    1.0980    1.5353   1.7378  1.8658];      % tumor volume in cm^3
survProb     = [1         1         0.6       0.4      0.2     0];      % survival probability at day t in control
survProb2     = [1         1         0.6       0.4      0.2     0.01];      % survival probability at day t in control


survParams_MLE_C = tumorMLE(survVol,survProb,ones(1,length(survProb)),1,@relDE_logistic,[3.3779 1.0184],[0 0.95]); % S(N) fitting
survGivenN_C = SofNeval(.112,survVol,survParams_MLE_C);
sN_C = SofNeval(.112,survVol,[4.7943    1.0040]);
%survParams_MLE_C = [4.1618 1.0120];


%% 


%%% Post IR
Time_IR   = [6       8       11      13      15      18      20];
TumVol_IR = [0.296   0.352   0.552   0.712   1.040   1.208   1.328];
TumSD_IR  = [0.176   0.176   0.232   0.304   0.328   0.340   0.360]; %% last two entries are chosen since only one surviving mouse

survTime_IR     = [6       7       11      14.5    17.5    20];      % days
survVol_IR      = [0.296   0.3581  0.552   0.8488  1.0983  1.3242];   % tumor volume in cm^3
survProb_IR     = [1       1       1       0.6     0.2     0];      % survival probability at day t in control

%survTime_IR     = [6       8       11      14      15      20];      % days
%survVol_IR      = [0.296   0.352   0.552   0.8138  1.040   1.328];   % tumor volume in cm^3
%survProb_IR     = [1       1       1       0.6     0.4     0];      % survival probability at day t in control

survParams_MLE_IR = tumorMLE(survVol_IR,survProb_IR,ones(1,length(survProb_IR)),1,@relDE_logistic,[8.2328 1.0065],[0 0.95]); % S(N) fitting
survGivenN_IR = SofNeval(.2928,survVol_IR,survParams_MLE_IR);
%survParams_MLE_IR = [5.6930 1.0719];

% PLOT THE DATA
figure(1)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
xlabel('Time in days')
ylabel('Tumor volume in cm^3')
box off

timdat2 = [1 4 7 10 14 17 21];
tuudata2 = [0.09420689655172418 0.12851724137931035 0.17197701149425293 0.26575862068965523 0.4007126436781609 0.5814137931034484 0.8421724137931036];
tuustd2 = [0.025 0.035 0.04 0.045 0.05 0.1 0.105];

% SET UP NINTIAL GUESS FOR LAMBDA, K AND A PROPOSAL FUNCTION

lam_MLE = 0.2782;
k_MLE =  4.77;
n0_MLE = .112; 

lam_MLE_IR = 0.1980;
k_MLE_IR =  3.9831;
n0_MLE_IR = .2928;

prop_sigmalam = 0.05;
prop_sigmaK = 0.5;
prop_sigmaN0 = 0.0880; 

prop_sigmalam_IR = 0.05;
prop_sigmaK_IR = 0.5;
prop_sigmaN0_IR = 0.176;

mu = [lam_MLE  k_MLE   n0_MLE];
sigma = [prop_sigmalam^2 0.0005 0.0005; 0.0005 prop_sigmaK^2 0.0005; 0.0005 0.0005 prop_sigmaN0^2];
R = chol(sigma);

mu_IR = [lam_MLE_IR  k_MLE_IR   n0_MLE_IR];
sigma_IR = [prop_sigmalam_IR^2 0.0005 0.0005; 0.0005 prop_sigmaK_IR^2 0.0005; 0.0005 0.0005 prop_sigmaN0_IR^2];
R_IR = chol(sigma_IR);


%% MH FOR TUMOR GROWTH PARAMS
Nrun = 50000;   % Number of iterations for Metropolis-Hastings


% SET UP INITIAL VECTOR OF LAMBDAS, IE THE MARKOV CHAIN
% these values will be updated but Matlab prefers to declare an array
% before the fact instead of it having to update the array length every
% time a new element is added 

lambda = zeros(1,Nrun);    % vector of 0's as the markov chain to be updated

ks = zeros(1,Nrun);

lambda(1) =  lam_MLE;   % This is the first element of the markov chain

ks(1) = k_MLE;

n0s = zeros(1,Nrun);

n0s(1) = n0_MLE;

lambda_IR = zeros(1,Nrun);    % vector of 0's as the markov chain to be updated

ks_IR = zeros(1,Nrun);

lambda_IR(1) =  lam_MLE_IR;   % This is the first element of the markov chain

ks_IR(1) = k_MLE_IR;

n0s_IR = zeros(1,Nrun);

n0s_IR(1) = n0_MLE_IR;

% SET UP INITIAL VECTOR OF ITERATION NUMBER
% these values will be updated but Matlab prefers to declare an array
% before the fact instead of it having to update the array length every
% time a new element is added. The iteration number will be used to make a
% trace plot

iter = zeros(1,Nrun);    % vector of 0's, will be updated

iter(1) =  1;   % This is the first iteration.


% RUN THE FOR LOOP TO GENERATE THE MARKOV CHAIN

for i = 2:1:Nrun  

    % disp(i)
    % Step 1: generate a random number from proposal distribution using the
    % previous state of the lambda Markov Chain. Here, we want a value
    % froma normal distribution with mean lambda(i-1) and std. dev.
    % prop_sigma.
    mucurr = [lambda(i-1) ks(i-1) n0s(i-1)];
    munew = mucurr + randn(1,3)*R;

    mucurr_IR = [lambda_IR(i-1) ks_IR(i-1) n0s_IR(i-1)];
    munew_IR = mucurr_IR + randn(1,3)*R_IR;

    lam_star = munew(1);
    k_star = munew(2);
    n0_star = munew(3);

    lam_star_IR = munew_IR(1);
    k_star_IR = munew_IR(2);
    n0_star_IR = munew_IR(3);


    % It is possible that lam_star is negative, so we want to preclude that
    if lam_star < 0
        lam_star = -lam_star;
    end
    % It is also possible that lam_star > 3, so we want to preclude that
    if lam_star > 0.75
        lam_star = 0.75 - (lam_star-0.75);
    end

    if k_star < 0
        k_star = -k_star;
    end

    if k_star > 10
        k_star = 10 - (k_star-10);
    end

    if n0_star <0
        n0_star = -n0_star;
    end

    if n0_star > 1 
        n0_star = 1 - (n0_star -1);
    end

    % It is possible that lam_star is negative, so we want to preclude that
    if lam_star_IR < 0
        lam_star_IR = -lam_star_IR;
    end
    % It is also possible that lam_star > 3, so we want to preclude that
    if lam_star_IR > 0.75
        lam_star_IR = 0.75 - (lam_star_IR-0.75);
    end

    if k_star_IR < 0
        k_star_IR = -k_star_IR;
    end

    if k_star_IR > 10
        k_star_IR = 10 - (k_star_IR-10);
    end

    if n0_star_IR <0
        n0_star_IR = 0.15+(0.15-n0_star_IR);
    end

    if n0_star_IR > 1 
        n0_star_IR = 1 - (n0_star_IR -1);
    end

    % Step 2: Compute acceptance probability using the formula from class
    % alpha = exp(0.5*(RSS(i-1) - RSS(*)).
    % for this we need weighted RSS at previous lambda and weighted RSS at
    % lam_star. I have defined a function RSSeval that evaluates RSS:
    
    %RSSprev = RSSeval(lambda(i-1),Time,VASs,VASsd);
    
    RSSprev = RSSeval(lambda(i-1), ks(i-1), n0s(i-1), Time, TumVol, TumSD);

    %RSSstar = RSSeval(lam_star,Time,VASs,VASsd);
    
    RSSstar = RSSeval(lam_star, k_star, n0_star, Time, TumVol, TumSD);

    alpha = exp(0.5*(RSSprev - RSSstar));

    %RSSprev = RSSeval(lambda(i-1),Time,VASs,VASsd);
    
    RSSprev_IR = RSSeval(lambda_IR(i-1), ks_IR(i-1), n0s_IR(i-1), Time_IR, TumVol_IR, TumSD_IR);

    %RSSstar = RSSeval(lam_star,Time,VASs,VASsd);
    
    RSSstar_IR = RSSeval(lam_star_IR, k_star_IR, n0_star_IR, Time_IR, TumVol_IR, TumSD_IR);

    alpha_IR = exp(0.5*(RSSprev_IR - RSSstar_IR));


    % Now throw an alpha-coin to decide whether to accept or reject lam_*
    % as the next state of the Markov Chain
    
    t = unifrnd(0,1);

    if t < alpha
       lambda(i) = lam_star;
       ks(i) = k_star;
       n0s(i) = n0_star;
    else
        lambda(i) = lambda(i-1);
        ks(i) = ks(i-1);
        n0s(i) = n0s(i-1);
    end

    if t < alpha_IR
       lambda_IR(i) = lam_star_IR;
       ks_IR(i) = k_star_IR;
       n0s_IR(i) = n0_star_IR;
    else
        lambda_IR(i) = lambda_IR(i-1);
        ks_IR(i) = ks_IR(i-1);
        n0s_IR(i) = n0s_IR(i-1);
    end

    clear lam_star k_star RSSstar RSSprev alpha mucurr munew lam_star_IR k_star_IR RSSstar_IR RSSprev_IR alpha_IR mucurr_IR munew_IR

    iter(i) = i;
    waitbar(i/Nrun)
end

%% INITIAL MH PARAM DISTS AND TRACE PLOTS
% figure(9)
% hold on
% hist3([lambda ; ks]');
% xlabel('\lambda');
% ylabel('K');
% zlabel('Frequency');
% title('3D Histogram of \lambda and K');
% set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica');
% box off;
% view(45, 30);
% 
% 
% figure(10)
% hold on
% set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica');
% hist3([lambda ; ks]','CdataMode','auto')
% xlabel('\lambda');
% ylabel('K');
% colorbar
% view(2)
% 
% figure(11)
% hold on
% set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica');
% hist3([lambda ; n0s]','CdataMode','auto')
% xlabel('\lambda');
% ylabel('N_0');
% colorbar
% view(2)
% 
% figure(12)
% hold on
% set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica');
% hist3([ks ; n0s]','CdataMode','auto')
% xlabel('K');
% ylabel('N_0');
% colorbar
% view(2)


figure(3)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
histogram(lambda, 'Normalization', 'probability')
histogram(lambda_IR, 'Normalization', 'probability')
xlabel('\lambda')
ylabel('probability')


figure(4)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
histogram(ks, 'Normalization', 'probability')
histogram(ks_IR, 'Normalization', 'probability')
xlabel('K')
ylabel('Probability')

figure(5)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
histogram(n0s, 'Normalization', 'probability')
histogram(n0s_IR, 'Normalization', 'probability')
xlabel('N_0')
ylabel('Probability')



figure(6)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
plot(lambda,iter,'k','LineWidth',2)
xlabel('\lambda')
ylabel('Iteration number')



figure(7)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
plot(ks,iter,'k','LineWidth',2)
xlabel('K')
ylabel('Iteration number')

figure(8)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
plot(n0s,iter,'k','LineWidth',2)
xlabel('N_0')
ylabel('Iteration number')

figure(56)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
plot(lambda_IR,iter,'k','LineWidth',2)
xlabel('\lambda')
ylabel('Iteration number')



figure(57)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
plot(ks_IR,iter,'k','LineWidth',2)
xlabel('K')
ylabel('Iteration number')

figure(58)
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
plot(n0s_IR,iter,'k','LineWidth',2)
xlabel('N_0')
ylabel('Iteration number')

%% PLOT INITIAL MH PARAM COMBINATION N(t) SOLUTIONS
%%%% Plot the results of the model accepted parameters

tspan = Time(1):1:Time(end);
tspan_IR = Time_IR(1):1:Time_IR(end);

Frun = zeros(length(tspan),Nrun);
Frun_IR = zeros(length(tspan_IR),Nrun);

for i = 1:1:Nrun

    Frun(:,i) = modeleval(lambda(i), ks(i), n0s(i), tspan);

end

for i = 1:1:Nrun

    Frun_IR(:,i) = modeleval(lambda_IR(i), ks_IR(i), n0s_IR(i), tspan_IR);

end

Frun = Frun';   % matlab mean computes mean of each column by default and not each row as we want
Favg = (mean(Frun))';
Fstd = (std(Frun))';

Frun_IR = Frun_IR';   % matlab mean computes mean of each column by default and not each row as we want
Favg_IR = (mean(Frun_IR))';
Fstd_IR = (std(Frun_IR))';

figure(1)
coord_up = [tspan',Favg+Fstd];
coord_low = [tspan',Favg-Fstd];
coord_combine = [coord_up;flipud(coord_low)];
fill(coord_combine(:,1),coord_combine(:,2),[0    0    1],'FaceAlpha',0.3,'EdgeColor','none')
errorbar(Time,TumVol,TumSD,'ko','MarkerSize',20,'LineWidth',2)
plot(tspan,Favg,'Color',[0    0    1],'LineWidth',3)
coord_up_IR = [tspan_IR',Favg_IR+Fstd_IR];
coord_low_IR = [tspan_IR',Favg_IR-Fstd_IR];
coord_combine_IR = [coord_up_IR;flipud(coord_low_IR)];
fill(coord_combine_IR(:,1),coord_combine_IR(:,2),[1    0    0],'FaceAlpha',0.3,'EdgeColor','none')
errorbar(Time_IR,TumVol_IR,TumSD_IR,'ko','MarkerSize',20,'LineWidth',2)
plot(tspan_IR,Favg_IR,'Color',[1    0    0],'LineWidth',3)

% figure(13)
% hold on
% set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
% histogram(Frun(:,end),'Normalization','pdf')
% xlabel('Predicted final tumor volume on day 55')
% ylabel('probability')
% 
% figure(132)
% hold on
% set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
% histogram(Frun_IR(:,end),'Normalization','pdf')
% xlabel('Predicted final tumor volume on day 55')
% ylabel('probability')

% %% 
% % Preallocate matrix for results
% sample_survParams_MLE_C = zeros(Nrun, 2);
% 
% % Loop over each sample
% for j = 1:Nrun
% 
%     % Sample tumor time and compute the tumor size at time t
%     sampleTumorTime_C = survTime;
%     sampleTumor_C = modeleval(lambda(j), ks(j), n0s(j), sampleTumorTime_C); % mouse j's tumor size at time t
% 
%     % Define the objective function for lsqcurvefit
%     objectiveFun = @(params, n) SofNeval(n0s(j), n, params) - survProb;
% 
%     % Define initial guess for parameters
%     initialParams = [3, 1.05]; % Modify as needed
% 
%     % Define lower and upper bounds for parameters
%     lb = [1, 0.95]; % Modify as needed
%     ub = [100, 2];   % Modify as needed
% 
%     % Perform parameter fitting
%     [pars, ~] = lsqcurvefit(objectiveFun, initialParams, sampleTumor_C, survProb, lb, ub);
% 
%     % Store the results
%     sample_survParams_MLE_C(j, :) = pars;
% 
% end


%% time of death scheme - IR AND CONTROL

Nrun2 = 100; % number of trials
%Nrun2 = 1000;
theta_C = zeros(Nrun2,Nrun); % time of death matrix, first entry is trial #, second is mouse #
theta_control = theta_C;
theta_outside = zeros(Nrun2,Nrun);
finalTumors_C = zeros(Nrun2, Nrun); % tumor size at time of death
finalTumors_outside = zeros(Nrun2, Nrun);
tspan_fine = Time(1):.01:18;
miceSurvival_C = zeros(Nrun, length(tspan_fine)); % fill with survival probability at time t for each mouse
miceSurvival_outside = zeros(Nrun, length(tspan_fine));

theta_IR = zeros(Nrun2,Nrun); % time of death matrix, first entry is trial #, second is mouse #
theta_treatment = theta_IR;
finalTumors_IR = zeros(Nrun2, Nrun); % tumor size at time of death
tspan_fine_IR = Time_IR(1):.01:23;
miceSurvival_IR = zeros(Nrun, length(tspan_fine_IR)); % fill with survival probability at time t for each mouse

SNparams_C = survParams_MLE_C;
SNparams_IR = survParams_MLE_IR;

%Msampler = [randperm(Nrun); randperm(Nrun); randperm(Nrun)]; % randomize sample number (for sampling parameters)
%outside_params = zeros(Nrun, 3);
%outside_sample_tumors = zeros(Nrun,length(tspan_fine));

accepted_lambda_C = zeros(Nrun,1);
accepted_k_C = zeros(Nrun,1);
accepted_n0_C = zeros(Nrun,1);
accepted_lambda_IR = zeros(Nrun,1);
accepted_k_IR = zeros(Nrun,1);
accepted_n0_IR = zeros(Nrun,1);

accepted_theta_C = zeros(Nrun,1);
accepted_theta_IR = zeros(Nrun, 1);
accepted_finalTumors_C = zeros(Nrun,1);
accepted_finalTumors_IR = zeros(Nrun,1);

for j=1:1:Nrun

    %aj = Msampler(1,j);
    %bj = Msampler(2,j);
    %cj = Msampler(3,j);

    %sampleTumor_outside = modeleval(lambda(aj), ks(bj), n0s(cj), tspan_fine); % mice outside experiment with no treatment
    %outside_params(j,:) = [lambda(aj) ks(bj) n0s(cj)];

    % Take in mice info
    sampleTumor_C = modeleval(lambda(j), ks(j), n0s(j), tspan_fine); % mouse j's tumor size at time t 
    sampleTumorTime_C = tspan_fine;

    % Take in mice info
    sampleTumor_IR = modeleval(lambda_IR(j), ks_IR(j), n0s_IR(j), tspan_fine_IR); % mouse j's tumor size at time t 
    sampleTumorTime_IR = tspan_fine_IR;

    % Compute mouse(j)'s tumor-survival probability
    sampleSNoutput_C = SofNeval(n0s(j),sampleTumor_C,SNparams_C);
    sampleSNoutput_IR = SofNeval(n0s_IR(j),sampleTumor_IR,SNparams_IR);

    % Compute mouse(j)'s time-survival probability
    miceSurvival_C(j,:) = sampleSNoutput_C';
    miceSurvival_IR(j,:) = sampleSNoutput_IR';

    for h=1:1:Nrun2

        coin = rand();
        idx = find(miceSurvival_C(j,:) <= coin, 1);
        
        idx_IR = find(miceSurvival_IR(j,:) <= coin, 1);

        if length(idx) == 0

            theta_C(h,j) = tspan_fine(end);
            finalTumors_C(h,j) = sampleTumor_C(end);

        else 

            theta_C(h,j) = tspan_fine(idx);
            finalTumors_C(h,j) = sampleTumor_C(idx);

        end

        if length(idx_IR) == 0

            theta_IR(h, j) = tspan_fine_IR(end);
            finalTumors_IR(h,j) = sampleTumor_IR(end);

        else

            theta_IR(h, j) = tspan_fine_IR(idx_IR);
            finalTumors_IR(h,j) = sampleTumor_IR(idx_IR);

        end

    end

    if median(theta_C(:,j))< 17

        accepted_lambda_C(j) = lambda(j);
        accepted_k_C(j) = ks(j);
        accepted_n0_C(j) = n0s(j);
        accepted_theta_C(j) = median(theta_C(:,j),1);
        accepted_finalTumors_C(j) = median(finalTumors_C(:,j),1);
        theta_control(:,j) = theta_C(:,j);

    end

    if median(theta_IR(:,j))< 22

        accepted_lambda_IR(j) = lambda_IR(j);
        accepted_k_IR(j) = ks_IR(j);
        accepted_n0_IR(j) = n0s_IR(j);
        accepted_theta_IR(j) = median(theta_IR(:,j),1);
        accepted_finalTumors_IR(j) = median(finalTumors_IR(:,j),1);
        theta_treatment(:,j) = theta_IR(:,j);


    end
    
    waitbar(j/Nrun)

end

% accepted_rho_C = sample_survParams_MLE_C(:,1);
% accepted_gamma_C = sample_survParams_MLE_C(:,2);
% accepted_rho_C = accepted_rho_C(accepted_lambda_C>0);
% accepted_gamma_C = accepted_gamma_C(accepted_lambda_C>0);
accepted_lambda_C = accepted_lambda_C(accepted_lambda_C>0);
accepted_k_C = accepted_k_C(accepted_k_C>0);
accepted_n0_C = accepted_n0_C(accepted_n0_C>0);
accepted_theta_C = accepted_theta_C(accepted_theta_C>0);
accepted_finalTumors_C = accepted_finalTumors_C(accepted_finalTumors_C>0);
accepted_lambda_IR = accepted_lambda_IR(accepted_lambda_IR>0);
accepted_k_IR = accepted_k_IR(accepted_k_IR>0);
accepted_n0_IR = accepted_n0_IR(accepted_n0_IR>0);
accepted_theta_IR = accepted_theta_IR(accepted_theta_IR>0);
accepted_finalTumors_IR = accepted_finalTumors_IR(accepted_finalTumors_IR>0);


%%  UPDATED OUTPUT WITH UPDATED PARAMETERS
tspan = Time(1):1:Time(end);
tspan_IR = Time_IR(1):1:Time_IR(end);

Frun_fin = zeros(length(tspan),length(accepted_lambda_C));
Frun_IR_fin = zeros(length(tspan_IR),length(accepted_lambda_IR));

for i = 1:1:length(accepted_lambda_C)

    Frun_fin(:,i) = modeleval(accepted_lambda_C(i), accepted_k_C(i), accepted_n0_C(i), tspan);

end

for i = 1:1:length(accepted_lambda_IR)

    Frun_IR_fin(:,i) = modeleval(accepted_lambda_IR(i), accepted_k_IR(i), accepted_n0_IR(i), tspan_IR);

end

Frun_fin = Frun_fin';   % matlab mean computes mean of each column by default and not each row as we want
Favg_fin = (mean(Frun_fin))';
Fstd_fin = (std(Frun_fin))';

Frun_IR_fin = Frun_IR_fin';   % matlab mean computes mean of each column by default and not each row as we want
Favg_IR_fin = (mean(Frun_IR_fin))';
Fstd_IR_fin = (std(Frun_IR_fin))';

figure(1)
hold on
coord_up_fin = [tspan',Favg_fin+Fstd_fin];
coord_low_fin = [tspan',Favg_fin-Fstd_fin];
coord_combine_fin = [coord_up_fin;flipud(coord_low_fin)];
fill(coord_combine_fin(:,1),coord_combine_fin(:,2),[0    1    1],'FaceAlpha',0.3,'EdgeColor','none')
errorbar(Time,TumVol,TumSD,'ko','MarkerSize',20,'LineWidth',2)
plot(tspan,Favg_fin,'Color',[0    1    1],'LineWidth',3)
coord_up_IR_fin = [tspan_IR',Favg_IR_fin+Fstd_IR_fin];
coord_low_IR_fin = [tspan_IR',Favg_IR_fin-Fstd_IR_fin];
coord_combine_IR_fin = [coord_up_IR_fin;flipud(coord_low_IR_fin)];
fill(coord_combine_IR_fin(:,1),coord_combine_IR_fin(:,2),[1   1   0],'FaceAlpha',0.3,'EdgeColor','none')
errorbar(Time_IR,TumVol_IR,TumSD_IR,'ko','MarkerSize',20,'LineWidth',2)
plot(tspan_IR,Favg_IR_fin,'Color',[1    1    0],'LineWidth',3)
hold off


%% time of death plots AND PARAMETER DISTS FOR MH PARAMS ASSOCIATED WITH APPROPRIATE TIMES OF DEATH

figure(3)
hold on
histogram(accepted_lambda_C, 'Normalization', 'probability');
histogram(accepted_lambda_IR, 'Normalization', 'probability');
title("Accepted \lambda for Control & IR Models")
hold off

figure(4)
hold on
histogram(accepted_k_C, 'Normalization', 'probability');
histogram(accepted_k_IR, 'Normalization', 'probability');
title("Accepted K for Control & IR Models")
hold off

figure(5)
hold on
histogram(Frun_fin(:,6), 'Normalization', 'probability');
histogram(accepted_n0_IR, 'Normalization', 'probability');
title("Accepted N6 for Control & IR Models")
hold off

accepted_n6_C = Frun_fin(:,6);

figure(14)
hold on
histogram(accepted_theta_C, 'Normalization', 'probability');
histogram(accepted_theta_IR, 'Normalization', 'probability');
title("Median Time of Death for Control and IR, \theta")
hold off

figure(15)
hold on
histogram(accepted_finalTumors_C, 'Normalization', 'probability');
histogram(accepted_finalTumors_IR, 'Normalization', 'probability');
title("Average Final Tumor Volume for Control & IR, N(\theta)")
hold off

% figure(20)
% hold on
% histogram(median(theta_outside,1), 'Normalization', 'probability');
% %histogram(median(theta_outside_IR,1), 'Normalization', 'probability');
% title("Median Time of Death for Control - outside of experiment, \theta_{outside}")
% hold off
% 
% figure(21)
% hold on
% histogram(mean(finalTumors_outside,1), 'Normalization', 'probability');
% %histogram(mean(finalTumors_outside_IR,1), 'Normalization', 'probability');
% title("Average Final Tumor Volume for Control - outside of experiment, N(\theta_{outside})")
% hold off

%% time of death scheme - OUTSIDE control

Nrun3 = length(accepted_lambda_C); 
Nrun4 = 100; % number of trials
theta_outside_C = zeros(Nrun4,Nrun3);
finalTumors_outside_C = zeros(Nrun4, Nrun3);
tspan_fine = Time(1):.01:40;
miceSurvival_outside_C = zeros(Nrun3, length(tspan_fine));

SNparams_C = survParams_MLE_C;

Msampler = [randperm(Nrun3); randperm(Nrun3); randperm(Nrun3); randperm(Nrun3); randperm(Nrun3)]; % randomize sample number (for sampling parameters)
outside_params_C = zeros(Nrun3, 3);
outside_sample_tumors = zeros(Nrun3,length(tspan_fine));

median_outside_theta_C = zeros(Nrun3,1);
median_outside_finalTumors_C = zeros(Nrun3,1);
outside_survivors_C = zeros(Nrun3,3);

for j=1:1:Nrun3

    aj = Msampler(1,j);
    bj = Msampler(2,j);
    cj = Msampler(3,j);
    dj = Msampler(4,j);
    ej = Msampler(5,j);

    sampleTumor_outside_C = modeleval(accepted_lambda_C(aj), accepted_k_C(bj), accepted_n0_C(cj), tspan_fine); % mice outside experiment with no treatment
    %sampleTumor_outside_C = modeleval(lambda(aj), ks(bj), n0s(cj), tspan_fine); % mice outside experiment with no treatment
    outside_sample_tumors(j,:) = sampleTumor_outside_C;
    outside_params_C(j,:) = [accepted_lambda_C(aj) accepted_k_C(bj) accepted_n0_C(cj)];
    %outside_params_C(j,:) = [lambda(aj) ks(bj) n0s(cj)];
    sampleTumorTime_outside_C = tspan_fine;

    % Compute mouse(j)'s tumor-survival probability
    sampleSNoutput_outside_C = SofNeval(accepted_n0_C(j),sampleTumor_outside_C,SNparams_C);
    %sampleSNoutput_outside_C = SofNeval(n0s(j),sampleTumor_outside_C,SNparams_C);

    % Compute mouse(j)'s time-survival probability
    miceSurvival_outside_C(j,:) = sampleSNoutput_outside_C';

    for h=1:1:Nrun4

        coin = rand();
        idxo = find(miceSurvival_outside_C(j,:) <= coin, 1);

        if length(idxo) ~= 0

            theta_outside_C(h, j) = sampleTumorTime_outside_C(idxo);
            finalTumors_outside_C(h,j) = sampleTumor_outside_C(idxo);

        else

            theta_outside_C(h,j) = 40;
            finalTumors_outside_C(h,j) = sampleTumor_outside_C(40);

        end

    end

    % if median(theta_outside_C(:,j),1)>35
    % 
    %     outside_survivors_C(j,:) = outside_params_C(j,:);
    % 
    % end
    % 
    % if median(theta_outside_C(:,j),1)<=35
    % 
    %     median_outside_theta_C(j) = median(theta_outside_C(:,j),1);
    %     median_outside_finalTumors_C(j) = median(finalTumors_outside_C(:,j),1);
    % 
    % else
    % 
    %     median_outside_theta_C(j) = 0;
    %     median_outside_finalTumors_C(j) = 0;
    %     outside_params_C(j,1) = 0;
    %     outside_params_C(j,2) = 0;
    %     outside_params_C(j,3) = 0;
    % 
    % end

    median_outside_theta_C(j) = median(theta_outside_C(:,j));
    median_outside_finalTumors_C(j) = median(finalTumors_outside_C(:,j));
    
    
    waitbar(j/Nrun3)

end

%% DATA FOR OUTSIDE CONTROL


median_outside_theta_C = median_outside_theta_C(median_outside_theta_C>0);
median_outside_finalTumors_C = median_outside_finalTumors_C(median_outside_finalTumors_C>0);
outside_lambda_C = outside_params_C(:,1);
outside_k_C = outside_params_C(:,2);
outside_n0_C = outside_params_C(:,3);
outside_lambda_C = outside_lambda_C(outside_lambda_C>0);
outside_k_C = outside_k_C(outside_k_C>0);
outside_n0_C = outside_n0_C(outside_n0_C>0);
% outside_survivors_lambda_C = outside_survivors_C(:,1);
% outside_survivors_k_C = outside_survivors_C(:,2);
% outside_survivors_n0_C = outside_survivors_C(:,3);
% outside_survivors_lambda_C = outside_survivors_lambda_C(outside_survivors_lambda_C>0);
% outside_survivors_k_C = outside_survivors_k_C(outside_survivors_k_C>0);
% outside_survivors_n0_C = outside_survivors_n0_C(outside_survivors_n0_C>0);

%% time of death scheme - OUTSIDE IR

Nrun5 = length(accepted_lambda_IR); 
Nrun4 = 50; % number of trials
theta_outside_IR = zeros(Nrun4,Nrun5);
finalTumors_outside_IR = zeros(Nrun4, Nrun5);
tspan_fine_IR = Time_IR(1):.01:Time_IR(end)+25;
miceSurvival_outside_IR = zeros(Nrun5, length(tspan_fine_IR));

SNparams_IR = survParams_MLE_IR;

Msampler = [randperm(Nrun5); randperm(Nrun5); randperm(Nrun5)]; % randomize sample number (for sampling parameters)
outside_params_IR = zeros(Nrun5, 3);
outside_sample_tumors_IR = zeros(Nrun5,length(tspan_fine_IR));

median_outside_theta_IR = zeros(Nrun5,1);
median_outside_finalTumors_IR = zeros(Nrun5,1);
outside_survivors_IR = zeros(Nrun5,3);

for j=1:1:Nrun5

    aj = Msampler(1,j);
    bj = Msampler(2,j);
    cj = Msampler(3,j);

    sampleTumor_outside_IR = modeleval(accepted_lambda_IR(aj), accepted_k_IR(bj), accepted_n0_IR(cj), tspan_fine_IR); % mice outside experiment with no treatment
    outside_params_IR(j,:) = [accepted_lambda_IR(aj) accepted_k_IR(bj) accepted_n0_IR(cj)];
    sampleTumorTime_outside_IR = tspan_fine_IR;

    % Compute mouse(j)'s tumor-survival probability
    sampleSNoutput_outside_IR = SofNeval(accepted_n0_IR(j),sampleTumor_outside_IR,SNparams_IR);

    % Compute mouse(j)'s time-survival probability
    miceSurvival_outside_IR(j,:) = sampleSNoutput_outside_IR';

    for h=1:1:Nrun4

        coin = rand();
        idxo = find(miceSurvival_outside_IR(j,:) <= coin, 1);

        if length(idxo) ~= 0

            theta_outside_IR(h, j) = sampleTumorTime_outside_IR(idxo);
            finalTumors_outside_IR(h,j) = sampleTumor_outside_IR(idxo);

        else

            theta_outside_IR(h,j) = -100;
            finalTumors_outside_IR(h,j) = -100;

        end

    end

    if mean(theta_outside_IR(:,j),1)<0

        outside_survivors_IR(j,:) = outside_params_IR(j,:);

    else

        outside_survivors_IR(j,:) = [-1 -1 -1];

    end
    
    if mean(theta_outside_IR(:,j),1)>0

        median_outside_theta_IR(j) = median(theta_outside_IR(:,j),1);
        median_outside_finalTumors_IR(j) = median(finalTumors_outside_IR(:,j),1);

    else

        median_outside_theta_IR(j) = -100;
        median_outside_finalTumors_IR(j) = -100;

    end
    
    
    waitbar(j/Nrun5)

end

median_outside_theta_IR = median_outside_theta_IR(median_outside_theta_IR>0);
median_outside_finalTumors_IR = median_outside_finalTumors_IR(median_outside_finalTumors_IR>0);
outside_survivors_lambda_IR = outside_survivors_IR(:,1);
outside_survivors_k_IR = outside_survivors_IR(:,2);
outside_survivors_n0_IR = outside_survivors_IR(:,3);
outside_survivors_lambda_IR = outside_survivors_lambda_IR(outside_survivors_lambda_IR>0);
outside_survivors_k_IR = outside_survivors_k_IR(outside_survivors_k_IR>0);
outside_survivors_n0_IR = outside_survivors_n0_IR(outside_survivors_n0_IR>0);

%% outside of experiment -- 'DETECTING GREENER PASTURES' (finding more preclinical trials)

%%%%%%%%%%%%%%%%%%% UPDATE FOR IR %%%%%%%%%%%%%%%%%%%%

desired_theta = 20; % set theta
desired_direction = 2; % to the left (1) of set theta or to the right (2)
if desired_direction <2

    xthetalocs=find(median_outside_theta_C<desired_theta);

else 

    xthetalocs=find(median_outside_theta_C>desired_theta);

end

xthetalocs = find(median_outside_theta_C>16 & median_outside_theta_C<34); % hopefully well behaved mice death range

outside_tumorsavg = mean(outside_sample_tumors);
outside_tumorsstd = std(outside_sample_tumors);

figure(22)
hold on
%%% Experiment data --  Control
coord_up = [tspan',Favg_fin+Fstd_fin];
coord_low = [tspan',Favg_fin-Fstd_fin];
coord_combine = [coord_up;flipud(coord_low)];
fill(coord_combine(:,1),coord_combine(:,2),[0    0    1],'FaceAlpha',0.3,'EdgeColor','none')
errorbar(Time,TumVol,TumSD,'ko','MarkerSize',20,'LineWidth',2)
plot(tspan,Favg_fin,'Color',[0    0    1],'LineWidth',3)
plot(tspan,Favg_fin+Fstd_fin)
plot(tspan,Favg_fin-Fstd_fin)
%%% Experiment data -- IR
coord_up_IR = [tspan_IR',Favg_IR_fin+Fstd_IR_fin];
coord_low_IR = [tspan_IR',Favg_IR_fin-Fstd_IR_fin];
coord_combine_IR = [coord_up_IR;flipud(coord_low_IR)];
fill(coord_combine_IR(:,1),coord_combine_IR(:,2),[1    0    0],'FaceAlpha',0.3,'EdgeColor','none')
errorbar(Time_IR,TumVol_IR,TumSD_IR,'ko','MarkerSize',20,'LineWidth',2)
plot(tspan_IR,Favg_IR_fin,'Color',[1    0    0],'LineWidth',3)
plot(tspan_IR,Favg_IR_fin+Fstd_IR_fin)
plot(tspan_IR,Favg_IR_fin-Fstd_IR_fin)
%%% Full data
coord_up2 = [tspan_fine' (outside_tumorsavg+outside_tumorsstd)'];
coord_low2 = [tspan_fine' (outside_tumorsavg-outside_tumorsstd)'];
coord_combine2 = [coord_up2; flipud(coord_low2)];
fill(coord_combine2(:,1),coord_combine2(:,2),[0    1    0],'FaceAlpha',0.1,'EdgeColor','none')
plot(tspan_fine,outside_tumorsavg,'Color',[0    1    0],'LineWidth',3)
plot(tspan_fine,outside_tumorsavg+outside_tumorsstd)
plot(tspan_fine,outside_tumorsavg-outside_tumorsstd)
%%% Outside experiment
coord_up3 = [tspan_fine' (mean(outside_sample_tumors(xthetalocs',:))+std(outside_sample_tumors(xthetalocs',:)))'];
coord_low3 = [tspan_fine' (mean(outside_sample_tumors(xthetalocs',:))-std(outside_sample_tumors(xthetalocs',:)))'];
coord_combine3 = [coord_up3; flipud(coord_low3)];
fill(coord_combine3(:,1),coord_combine3(:,2),[1    0    1],'FaceAlpha',0.1,'EdgeColor','none')
plot(tspan_fine,mean(outside_sample_tumors(xthetalocs',:)),'Color',[1    0    1],'LineWidth',3)
plot(tspan_fine,mean(outside_sample_tumors(xthetalocs',:))+std(outside_sample_tumors(xthetalocs',:)))
plot(tspan_fine,mean(outside_sample_tumors(xthetalocs',:))-std(outside_sample_tumors(xthetalocs',:)))
%xline(desired_theta, '--r', '\theta wall') % where set time of death is
plot(timdat2, tuudata2,'LineWidth',3)
errorbar(timdat2, tuudata2, tuustd2, 'ko', 'MarkerSize',20,'LineWidth',2)
hold off

%% ACCEPTED PARAMETER DISTRIBUTION AND OUTSIDE PARAMETER DISTRIBUTION


figure(23) %%% population lambda dist (blue), IR K dist (red), and outside lambda dist
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
histogram(accepted_lambda_C,'Normalization', 'probability')
histogram(accepted_lambda_IR,'Normalization', 'probability')
histogram(outside_lambda_C(xthetalocs'),'Normalization', 'probability')
xlabel('\lambda')
ylabel('Probability')
hold off

figure(24) %%% population K dist (blue), IR K dist (red), and outside K dist
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
histogram(accepted_k_C,'Normalization', 'probability')
histogram(accepted_k_IR,'Normalization', 'probability')
histogram(outside_k_C(xthetalocs'),'Normalization', 'probability')
xlabel('K')
ylabel('Probability')
hold off

figure(25) %%% population N0 dist (blue), N0 IR dist (red), and outside N0 dist
hold on
set(gca,'LineWidth',2,'FontSize',30,'FontWeight','normal','FontName','Helvetica')
histogram(accepted_n0_C,'Normalization', 'probability')
histogram(accepted_n0_IR,'Normalization', 'probability')
histogram(outside_n0_C(xthetalocs'),'Normalization', 'probability')
histogram(Frun_fin(:,6),'Normalization', 'probability')
xlabel('N_0')
ylabel('Probability')
hold off

%% REDEFINE DATA TO FIT THE HOPEFULLY WELL BEHAVED MICE DATA

outtums = outside_sample_tumors(xthetalocs',:);

%figure(1)
%plot(timdat2, tuudata2,'LineWidth',3)
%hold on
%errorbar(timdat2, tuudata2, tuustd2, 'ko', 'MarkerSize',20,'LineWidth',2)

outtums_vals = zeros(length(outtums),7);
vals = [1 301 601 901 1301 1601 2001];

for i=1:1:length(outside_lambda_C(xthetalocs'))

    for j=1:1:7

        outtums_vals(i,j) = outtums(i,vals(j));
    
    end

end

%% FIND MICE FROM HOPEFULLY WELL BEHAVED MICE DATA


uppies = tuudata2+tuustd2;
downzo = tuudata2-tuustd2;
countu = zeros(length(outside_lambda_C(xthetalocs)),1);
countd = zeros(length(outside_lambda_C(xthetalocs)),1);
blip = zeros(length(outside_lambda_C(xthetalocs)),7);

for i=1:1:length(outside_lambda_C(xthetalocs))

    upco = uppies-outtums_vals(i,:);
    downco = outtums_vals(i,:)-downzo;
    blip(i,:) = abs(outtums_vals(i,:)-tuudata2);
    countu(i) = length(nonzeros(upco(upco>0)));
    countd(i) = length(nonzeros(downco(downco>0)));

end

counties = [countu countd];
ABCD=[tuustd2(1)*ones(1,length(outside_lambda_C(xthetalocs))); tuustd2(2)*ones(1,length(outside_lambda_C(xthetalocs))); tuustd2(3)*ones(1,length(outside_lambda_C(xthetalocs))); tuustd2(4)*ones(1,length(outside_lambda_C(xthetalocs))); tuustd2(5)*ones(1,length(outside_lambda_C(xthetalocs))); tuustd2(6)*ones(1,length(outside_lambda_C(xthetalocs))); tuustd2(7)*ones(1,length(outside_lambda_C(xthetalocs)))]';
XYZ = sum(blip<ABCD,2)>6;
successfully_recovered_tumors = zeros(length(outside_lambda_C(xthetalocs)),length(tspan_fine));
truncated_srts = zeros(length(outside_lambda_C(xthetalocs)),7);
lamso = outside_lambda_C(xthetalocs);
kso = outside_k_C(xthetalocs);
n0so = outside_n0_C(xthetalocs);
thetaso = median_outside_theta_C(xthetalocs);
wout_params = [lamso kso n0so thetaso];
thetasout = theta_outside_C(:,xthetalocs);
srts_params = zeros(length(lamso),4);
suslocs = zeros(length(outside_lambda_C(xthetalocs)));
srts_theta_full = zeros(100,length(lamso));

for i=1:1:length(outside_lambda_C(xthetalocs))

    if XYZ(i) == 1

        successfully_recovered_tumors(i,:) = outtums(i,:);
        suslocs(i) = i;
        srts_params(i,:) = [lamso(i) kso(i) n0so(i) thetaso(i)];
        srts_thata_full(:,i) = thetasout(:,i);
        truncated_srts(i,:) = outtums_vals(i,:);
   

    end

end

srts_lambda = srts_params(:,1);
srts_k = srts_params(:,2);
srts_n0 = srts_params(:,3);
srts_theta = srts_params(:,4);
srts_lambda = srts_lambda(srts_lambda>0);
srts_k = srts_k(srts_k>0);
srts_n0 = srts_n0(srts_n0>0);
srts_theta = srts_theta(srts_theta>0);
srts_params = [srts_lambda srts_k srts_n0 srts_theta];
successfully_recovered_tumors( ~any(successfully_recovered_tumors,2), : ) = [];
truncated_srts( ~any(truncated_srts,2), : ) = [];
thetasout = thetasout(:,suslocs(suslocs>0));

%% SUCCESSFULLY RECOVERED TUMORS VOLS AND KMS
% hopefully well behaved mice KM
NDKMx = [1 18 18 21 21 23 23 32 32];
NDKMy = [1 1 .9 .9 .7 .7 .1 .1 0];
kmavg = mean(thetasout);
upperkmout=mean(thetasout)+std(thetasout);
lowerkmout=mean(thetasout)-std(thetasout);

srtsKMx1 = 14:39;
srtsKMproba = zeros(1,length(srtsKMx1));
srtsKMprobl = zeros(1,length(srtsKMx1));
srtsKMprobu = zeros(1,length(srtsKMx1));

for i = 1:length(srtsKMx1)
    srtsKMproba(i) = (length(kmavg)-sum(kmavg<srtsKMx1(i)))/length(kmavg);
    srtsKMprobl(i) = (length(lowerkmout)-sum(lowerkmout<srtsKMx1(i)))/length(lowerkmout);
    srtsKMprobu(i) = (length(upperkmout)-sum(upperkmout<srtsKMx1(i)))/length(upperkmout);
end

srtsKMx2 = [1 srtsKMx1 srtsKMx1];
srtsKMy2 = [1 srtsKMproba srtsKMproba];
srtsKMy2l = [1 srtsKMprobl srtsKMprobl];
srtsKMy2u = [1 srtsKMprobu srtsKMprobu];


N = numel(srtsKMx2);
ia = 1+rem(0:N-1, N);
srtsKMx3 = [srtsKMx2(ia) ; srtsKMx2(ia)];
srtsKMx3 = srtsKMx3';
j=1;
srtsKMx = ones(1,2*N);
for i = 2:2*N

    if mod(i,2)==0
       j=i/2;
       srtsKMx(i) = srtsKMx3(j,1);
    else
        j=(i-1)/2;
        srtsKMx(i) = srtsKMx3(j,2);
    end


end

srtsKMx = srtsKMx(1,3:N);

N = numel(srtsKMy2);
ia = 1+rem(0:N-1, N);
srtsKMy3 = [srtsKMy2(ia) ; srtsKMy2(ia)];
srtsKMy3 = srtsKMy3';
j=1;
srtsKMy = ones(1,2*N);
for i = 2:2*N

    if mod(i,2)==0
       j=i/2;
       srtsKMy(i) = srtsKMy3(j,1);
    else
        j=(i-1)/2;
        srtsKMy(i) = srtsKMy3(j,2);
    end


end

N = numel(srtsKMy2l);
ia = 1+rem(0:N-1, N);
srtsKMy3 = [srtsKMy2l(ia) ; srtsKMy2l(ia)];
srtsKMy3 = srtsKMy3';
j=1;
srtsKMyl = ones(1,2*N);
for i = 2:2*N

    if mod(i,2)==0
       j=i/2;
       srtsKMyl(i) = srtsKMy3(j,1);
    else
        j=(i-1)/2;
        srtsKMyl(i) = srtsKMy3(j,2);
    end


end

N = numel(srtsKMy2u);
ia = 1+rem(0:N-1, N);
srtsKMy3 = [srtsKMy2u(ia) ; srtsKMy2u(ia)];
srtsKMy3 = srtsKMy3';
j=1;
srtsKMyu = ones(1,2*N);
for i = 2:2*N

    if mod(i,2)==0
       j=i/2;
       srtsKMyu(i) = srtsKMy3(j,1);
    else
        j=(i-1)/2;
        srtsKMyu(i) = srtsKMy3(j,2);
    end


end

srtsKMy = [srtsKMy(1,4:N) 0];
srtsKMyl = [srtsKMyl(1,4:N) 0];
srtsKMyu = [srtsKMyu(1,4:N) 0];



%% PLOT TUMORS THROUGH HOPEFULLY WELL BEHAVED MICE

figure
hold on
plot(timdat2,mean(truncated_srts,1),'Color',[1 0 1])
plot(timdat2,mean(truncated_srts,1)+std(truncated_srts),'Color',[1 0 1])
plot(timdat2,mean(truncated_srts,1)-std(truncated_srts),'Color',[1 0 1])
coord_upN = [timdat2' (mean(truncated_srts,1)+std(truncated_srts))'];
coord_lowN = [timdat2' (mean(truncated_srts,1)-std(truncated_srts))'];
coord_combineN = [coord_upN; flipud(coord_lowN)];
fill(coord_combineN(:,1),coord_combineN(:,2),[1    0    1],'FaceAlpha',0.1,'EdgeColor','none')
plot(timdat2, tuudata2,'LineWidth',3)
errorbar(timdat2, tuudata2, tuustd2, 'ko', 'MarkerSize',20,'LineWidth',2)
hold off

figure
hold on
plot(NDKMx,NDKMy)
plot(srtsKMx,srtsKMy)
coord_upKM = [srtsKMx' srtsKMyu'];
coord_lowKM = [srtsKMx' srtsKMyl'];
coord_combineKM = [coord_upKM; flipud(coord_lowKM)];
fill(coord_combineKM(:,1),coord_combineKM(:,2),[1    0.5    0.5],'FaceAlpha',0.1,'EdgeColor','none') %%% Plot fill mean +- std
hold off
% %% 
% Suslocsf = find(suslocs>.5);
% thetaconv = round(srts_theta*100);
% fataltummie = zeros(length(Suslocsf),1);
% for i=1:length(Suslocsf)
%     locis = Suslocsf(i);
%     loco = thetaconv(i);
%     fataltummie(i) = outtums(locis,loco);
% end

%% FULL WHOS OUT KM COMP

kmavg = mean(theta_outside_C);
upperkmout=mean(theta_outside_C)+std(theta_outside_C);
lowerkmout=mean(theta_outside_C)-std(theta_outside_C);

srtsKMx1 = 1:40;
srtsKMproba = zeros(1,length(srtsKMx1));
srtsKMprobl = zeros(1,length(srtsKMx1));
srtsKMprobu = zeros(1,length(srtsKMx1));

for i = 1:length(srtsKMx1)
    srtsKMproba(i) = (length(kmavg)-sum(kmavg<srtsKMx1(i)))/length(kmavg);
    srtsKMprobl(i) = (length(lowerkmout)-sum(lowerkmout<srtsKMx1(i)))/length(lowerkmout);
    srtsKMprobu(i) = (length(upperkmout)-sum(upperkmout<srtsKMx1(i)))/length(upperkmout);
end

srtsKMx2 = [1 srtsKMx1 srtsKMx1];
srtsKMy2 = [1 srtsKMproba srtsKMproba];
srtsKMy2l = [1 srtsKMprobl srtsKMprobl];
srtsKMy2u = [1 srtsKMprobu srtsKMprobu];


N = numel(srtsKMx2);
ia = 1+rem(0:N-1, N);
srtsKMx3 = [srtsKMx2(ia) ; srtsKMx2(ia)];
srtsKMx3 = srtsKMx3';
j=1;
srtsKMx = ones(1,2*N);
for i = 2:2*N

    if mod(i,2)==0
       j=i/2;
       srtsKMx(i) = srtsKMx3(j,1);
    else
        j=(i-1)/2;
        srtsKMx(i) = srtsKMx3(j,2);
    end


end

srtsKMx = srtsKMx(1,3:N);

N = numel(srtsKMy2);
ia = 1+rem(0:N-1, N);
srtsKMy3 = [srtsKMy2(ia) ; srtsKMy2(ia)];
srtsKMy3 = srtsKMy3';
j=1;
srtsKMy = ones(1,2*N);
for i = 2:2*N

    if mod(i,2)==0
       j=i/2;
       srtsKMy(i) = srtsKMy3(j,1);
    else
        j=(i-1)/2;
        srtsKMy(i) = srtsKMy3(j,2);
    end


end

N = numel(srtsKMy2l);
ia = 1+rem(0:N-1, N);
srtsKMy3 = [srtsKMy2l(ia) ; srtsKMy2l(ia)];
srtsKMy3 = srtsKMy3';
j=1;
srtsKMyl = ones(1,2*N);
for i = 2:2*N

    if mod(i,2)==0
       j=i/2;
       srtsKMyl(i) = srtsKMy3(j,1);
    else
        j=(i-1)/2;
        srtsKMyl(i) = srtsKMy3(j,2);
    end


end

N = numel(srtsKMy2u);
ia = 1+rem(0:N-1, N);
srtsKMy3 = [srtsKMy2u(ia) ; srtsKMy2u(ia)];
srtsKMy3 = srtsKMy3';
j=1;
srtsKMyu = ones(1,2*N);
for i = 2:2*N

    if mod(i,2)==0
       j=i/2;
       srtsKMyu(i) = srtsKMy3(j,1);
    else
        j=(i-1)/2;
        srtsKMyu(i) = srtsKMy3(j,2);
    end


end

srtsKMy = [srtsKMy(1,4:N) 0];
srtsKMyl = [srtsKMyl(1,4:N) 0];
srtsKMyu = [srtsKMyu(1,4:N) 0];

%% 
figure
hold on
plot(srtsKMx,srtsKMy)
coord_upKM = [srtsKMx' srtsKMyu'];
coord_lowKM = [srtsKMx' srtsKMyl'];
coord_combineKM = [coord_upKM; flipud(coord_lowKM)];
fill(coord_combineKM(:,1),coord_combineKM(:,2),[0.5    0.7    1],'FaceAlpha',0.1,'EdgeColor','none') %%% Plot fill mean +- std
plot(NDKMx,NDKMy)
xline(16)
xline(34)
hold off



%% 
NDKMx = [1 18 18 21 21 23 23 32 32];
NDKMy = [1 1 .9 .9 .7 .7 .1 .1 0];
srtsKMx1 = 17:33;


x1=[18 21 23 32];
y1=[.9 .7 .1 0];
x2=srtsKMx1(8:end);
y2=srtsKMprob(8:end);

% Combine unique time points from both groups
combined_times = unique([x1, x2]);

% Interpolate survival proportions to the combined time points
y1_interp = interp1(x1, y1, combined_times, 'previous', 'extrap');
y2_interp = interp1(x2, y2, combined_times, 'previous', 'extrap');

% Assume initial group sizes
n1 = 10; % Number of participants in group 1
n2 = 48; % Number of participants in group 2

% Calculate deaths (events) at each time point
events1 = [1 0 0 3 0 9 0 0 0 0 0 0 0 0 10 0]; % Deaths for group 1
events2 = [0 0 0 0 0 0 2 9 14 23 35 37 40 44 47 48]; % Deaths for group 2

% Calculate at-risk populations
at_risk1 = n1-events1;
at_risk2 = n2 - events2;

% Log-rank test calculations
observed = events1 + events2; % Total observed events at each time
expected1 = observed .* (at_risk1 ./ (at_risk1 + at_risk2));
expected2 = observed .* (at_risk2 ./ (at_risk1 + at_risk2));
valid = (expected1 + expected2) > 0;

% Compute log-rank statistic
log_rank_stat = sum(((events1(valid) - expected1(valid)).^2) ./ ...
                    (expected1(valid) + expected2(valid)));

% Degrees of freedom (1 for two groups)
df = 1;

% Compute p-value
p_value = 1 - chi2cdf(log_rank_stat, df);

% Display results
fprintf('Log-rank statistic: %.3f\n', log_rank_stat);
fprintf('P-value: %.3f\n', p_value);


%% KAPLAN-MEIER CURVE FIT

%%% Compute mean & std of times of death %%%
KMtheta_C_avg = mean(theta_C);
KMtheta_C_std = std(theta_C);
KMtheta_CL = KMtheta_C_avg-KMtheta_C_std;
KMtheta_CU = KMtheta_C_avg+KMtheta_C_std;
%%%

%%% Find total deaths (mean) each day for KM %%%
KMthetalocs1 = find(KMtheta_C_avg>8 & KMtheta_C_avg<9); % locations of deaths on day 8
KMy1 = length(KMthetalocs1); % number that died during day 8
KMthetalocs2 = find(KMtheta_C_avg>9 & KMtheta_C_avg<10);
KMy2 = length(KMthetalocs2);
KMthetalocs3 = find(KMtheta_C_avg>10 & KMtheta_C_avg<11);
KMy3 = length(KMthetalocs3);
KMthetalocs4 = find(KMtheta_C_avg>11 & KMtheta_C_avg<12);
KMy4 = length(KMthetalocs4);
KMthetalocs5 = find(KMtheta_C_avg>12 & KMtheta_C_avg<13);
KMy5 = length(KMthetalocs5);
KMthetalocs6 = find(KMtheta_C_avg>13 & KMtheta_C_avg<14);
KMy6 = length(KMthetalocs6);
KMthetalocs7 = find(KMtheta_C_avg>14 & KMtheta_C_avg<15);
KMy7 = length(KMthetalocs7);
KMthetalocs8 = find(KMtheta_C_avg>15 & KMtheta_C_avg<16);
KMy8 = length(KMthetalocs8);
KMthetalocs9 = find(KMtheta_C_avg>16 & KMtheta_C_avg<17);
KMy9 = length(KMthetalocs9);
KMthetalocs10 = find(KMtheta_C_avg>17);
KMy10 = length(KMthetalocs10);
KMtot = KMy1+KMy2+KMy3+KMy4+KMy5+KMy6+KMy7+KMy8+KMy9+KMy10; % total number that died between day 8 and 17.5
%%%

%%% Find total deaths (mean-std) each day for KM %%%
KMthetalocs1L = find(KMtheta_CL>8 & KMtheta_CL<9);
KMy1L = length(KMthetalocs1L);
KMthetalocs2L = find(KMtheta_CL>9 & KMtheta_CL<10);
KMy2L = length(KMthetalocs2L);
KMthetalocs3L = find(KMtheta_CL>10 & KMtheta_CL<11);
KMy3L = length(KMthetalocs3L);
KMthetalocs4L = find(KMtheta_CL>11 & KMtheta_CL<12);
KMy4L = length(KMthetalocs4L);
KMthetalocs5L = find(KMtheta_CL>12 & KMtheta_CL<13);
KMy5L = length(KMthetalocs5L);
KMthetalocs6L = find(KMtheta_CL>13 & KMtheta_CL<14);
KMy6L = length(KMthetalocs6L);
KMthetalocs7L = find(KMtheta_CL>14 & KMtheta_CL<15);
KMy7L = length(KMthetalocs7L);
KMthetalocs8L = find(KMtheta_CL>15 & KMtheta_CL<16);
KMy8L = length(KMthetalocs8L);
KMthetalocs9L = find(KMtheta_CL>16 & KMtheta_CL<17);
KMy9L = length(KMthetalocs9L);
KMthetalocs10L = find(KMtheta_CL>17);
KMy10L = length(KMthetalocs10L);
KMtotL = KMy1L+KMy2L+KMy3L+KMy4L+KMy5L+KMy6L+KMy7L+KMy8L+KMy9L+KMy10L;
%%%

%%% Find total deaths (mean+std) each day for KM %%%
KMthetalocs1U = find(KMtheta_CU>8 & KMtheta_CU<9);
KMy1U = length(KMthetalocs1U);
KMthetalocs2U = find(KMtheta_CU>9 & KMtheta_CU<10);
KMy2U = length(KMthetalocs2U);
KMthetalocs3U = find(KMtheta_CU>10 & KMtheta_CU<11);
KMy3U = length(KMthetalocs3U);
KMthetalocs4U = find(KMtheta_CU>11 & KMtheta_CU<12);
KMy4U = length(KMthetalocs4U);
KMthetalocs5U = find(KMtheta_CU>12 & KMtheta_CU<13);
KMy5U = length(KMthetalocs5U);
KMthetalocs6U = find(KMtheta_CU>13 & KMtheta_CU<14);
KMy6U = length(KMthetalocs6U);
KMthetalocs7U = find(KMtheta_CU>14 & KMtheta_CU<15);
KMy7U = length(KMthetalocs7U);
KMthetalocs8U = find(KMtheta_CU>15 & KMtheta_CU<16);
KMy8U = length(KMthetalocs8U);
KMthetalocs9U = find(KMtheta_CU>16 & KMtheta_CU<17);
KMy9U = length(KMthetalocs9U);
KMthetalocs10U = find(KMtheta_CU>17);
KMy10U = length(KMthetalocs10U);
KMtotU = KMy1U+KMy2U+KMy3U+KMy4U+KMy5U+KMy6U+KMy7U+KMy8U+KMy9U+KMy10U;
%%%

%%% Mean KM %%%
KMx = [1 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17]; % KM rectangle x verticies
KMy = [KMtot/KMtot KMtot/KMtot (KMtot--KMy1)/KMtot (KMtot-KMy1)/KMtot (KMtot-KMy1-KMy2)/KMtot (KMtot-KMy1-KMy2)/KMtot (KMtot-KMy1-KMy2-KMy3)/KMtot (KMtot-KMy1-KMy2-KMy3)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6-KMy7)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6-KMy7)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6-KMy7-KMy8)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6-KMy7-KMy8)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6-KMy7-KMy8-KMy9)/KMtot (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6-KMy7-KMy8-KMy9)/KMtot 0]; % (KMtot-KMy1-KMy2-KMy3-KMy4-KMy5-KMy6-KMy7-KMy8-KMy9-KMy10)/KMtot
%%%

%%% Mean KM - std %%%
KMxL = [1 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17]; % KM rectangle x verticies
KMyL = [KMtotL/KMtotL KMtotL/KMtotL (KMtotL-KMy1L)/KMtotL (KMtotL-KMy1L)/KMtotL (KMtotL-KMy1L-KMy2L)/KMtotL (KMtotL-KMy1L-KMy2L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L-KMy7L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L-KMy7L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L-KMy7L-KMy8L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L-KMy7L-KMy8L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L-KMy7L-KMy8L-KMy9L)/KMtotL (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L-KMy7L-KMy8L-KMy9L)/KMtotL 0]; % (KMtotL-KMy1L-KMy2L-KMy3L-KMy4L-KMy5L-KMy6L-KMy7L-KMy8L-KMy9L-KMy10L)/KMtotL
%%%

%%% Mean KM + std %%%
KMxU = [1 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 16 16 17 17];
KMyU = [KMtotU/KMtotU KMtotU/KMtotU (KMtotU-KMy1U)/KMtotU (KMtotU-KMy1U)/KMtotU (KMtotU-KMy1U-KMy2U)/KMtotU (KMtotU-KMy1U-KMy2U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U-KMy7U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U-KMy7U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U-KMy7U-KMy8U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U-KMy7U-KMy8U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U-KMy7U-KMy8U-KMy9U)/KMtotU (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U-KMy7U-KMy8U-KMy9U)/KMtotU 0]; % (KMtotU-KMy1U-KMy2U-KMy3U-KMy4U-KMy5U-KMy6U-KMy7U-KMy8U-KMy9U-KMy10U)/KMtotU
%%%

%%% KM Data %%%
KMyC = [1 1 .6 .6 .4 .4 .2 .2 0 0];
KMxC = [1 10 10 13 13 14 14 15 15 16];
%%%

%%% Plot %%%
plot(KMx,KMy, '-o','LineWidth',3,'Color',[0.5 0.7 1]) %%% Plot mean KM
hold on
coord_upKM = [KMx' KMyU'];
coord_lowKM = [KMx' KMyL'];
coord_combineKM = [coord_upKM; flipud(coord_lowKM)];
fill(coord_combineKM(:,1),coord_combineKM(:,2),[0.5    0.7    1],'FaceAlpha',0.1,'EdgeColor','none') %%% Plot fill mean +- std
plot(KMxC,KMyC, '-o','LineWidth',3,'Color',[0 0 1]) %%% Plot KM data
hold off
%%%

%% WEBS FOR CONTROL

min_k = min(min(accepted_k_C),min(accepted_k_IR));
min_lambda = min(min(accepted_lambda_C),min(accepted_lambda_IR));
min_n0 = min(min(accepted_n6_C),min(accepted_n0_IR));
min_n6 = min(min(accepted_n6_C),min(accepted_n0_IR));
min_theta = min(min(accepted_theta_C),min(accepted_theta_IR));

max_k = max(max(accepted_k_C),max(accepted_k_IR));
max_lambda = max(max(accepted_lambda_C),max(accepted_lambda_IR));
max_n6 = max(max(accepted_n6_C),max(accepted_n0_IR));
max_theta = max(max(accepted_theta_C),max(accepted_theta_IR));


normalized_k_C = (accepted_k_C-min_k*ones(length(accepted_k_C),1))/(max_k-min_k);
normalized_lambda_C = (accepted_lambda_C-min_lambda*ones(length(accepted_lambda_C),1))/(max_lambda-min_lambda);
normalized_n0_C = (accepted_n0_C-min_n0*ones(length(accepted_n0_C),1))/(max_n6-min_n0);
normalized_theta_C = (accepted_theta_C-min_theta*ones(length(accepted_theta_C),1))/(max_theta-min_theta);
normalized_n6_C = (accepted_n6_C-min_n6*ones(length(accepted_n6_C),1))/(max_n6-min_n6);

L10locs = find(normalized_theta_C<0.12);
lower_10_lambda = normalized_lambda_C(L10locs);
lower_10_k = normalized_k_C(L10locs);
lower_10_n0 = normalized_n0_C(L10locs);
lower_10_theta = normalized_theta_C(L10locs);

U10locs = find(normalized_theta_C>0.99);
upper_10_lambda = normalized_lambda_C(U10locs);
upper_10_k = normalized_k_C(U10locs);
upper_10_n0 = normalized_n0_C(U10locs);
upper_10_theta = normalized_theta_C(U10locs);

parameters = {'\lambda', 'K', 'N0', '\theta'};
values = [normalized_lambda_C normalized_k_C normalized_n0_C normalized_theta_C]';
valuesU = [upper_10_lambda upper_10_k upper_10_n0 upper_10_theta]';
valuesL = [lower_10_lambda lower_10_k lower_10_n0 lower_10_theta]';
%values = [[1, 2, 3, 4]; [4, 3, 2, 1]]'; % example 

%% 


figure;
hold on
plot(values, '-o','Color',[0 0 0]);
plot(valuesU, '-o','Color',[0 0 1]);
plot(valuesL, '-o','Color',[1 0 0]);
set(gca, 'XTick', 1:length(parameters), 'XTickLabel', parameters);
xlabel('Parameter');
ylabel('Value');
title('Normalized tumor growth parameter combinations vs \theta');
fontsize(18,"points")
grid on;
hold off

%% WEBS FOR TREATMENT

normalized_k_IR = (accepted_k_IR-min_k*ones(length(accepted_k_IR),1))/(max_k-min_k);
normalized_lambda_IR = (accepted_lambda_IR-min_lambda*ones(length(accepted_lambda_IR),1))/(max_lambda-min_lambda);
normalized_n0_IR = (accepted_n0_IR-min_n6*ones(length(accepted_n0_IR),1))/(max_n6-min_n6);
normalized_theta_IR = (accepted_theta_IR-min_theta*ones(length(accepted_theta_IR),1))/(max_theta-min_theta);

L10locs2 = find(normalized_theta_IR<0.075);
lower_10_lambdaIR = normalized_lambda_IR(L10locs2);
lower_10_kIR = normalized_k_IR(L10locs2);
lower_10_n0IR = normalized_n0_IR(L10locs2);
lower_10_thetaIR = normalized_theta_IR(L10locs2);

U10locs2 = find(normalized_theta_IR>0.99);
upper_10_lambdaIR = normalized_lambda_IR(U10locs2);
upper_10_kIR = normalized_k_IR(U10locs2);
upper_10_n0IR = normalized_n0_IR(U10locs2);
upper_10_thetaIR = normalized_theta_IR(U10locs2);

parameters2 = {'\lambda', 'K', 'N6', '\theta'};
values2 = [normalized_lambda_IR normalized_k_IR normalized_n0_IR normalized_theta_IR]';
valuesU2 = [upper_10_lambdaIR upper_10_kIR upper_10_n0IR upper_10_thetaIR]';
valuesL2 = [lower_10_lambdaIR lower_10_kIR lower_10_n0IR lower_10_thetaIR]';
%values = [[1, 2, 3, 4]; [4, 3, 2, 1]]'; % example 
%% 


figure;
hold on
plot(values2, '-o','Color',[0.7 0.7 0.7]);
plot(valuesU2, '-o','Color',[0 0 1]);
plot(valuesL2, '-o','Color',[1 0 0]);
set(gca, 'XTick', 1:length(parameters2), 'XTickLabel', parameters2);
xlabel('Parameter');
ylabel('Value');
title('Normalized post-IR tumor growth parameter combinations vs \theta');
fontsize(18,"points")
grid on;
hold off


%% 






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Fout = RSSeval(lam, k, inttum, Time, TumVol, TumSD)

% options = odeset('RelTol',1e-5,'AbsTol',1e-7);


F = inttum*k./(inttum + (k-inttum)*exp(-lam*(Time-Time(1))));

SQ_Res = norm((TumVol - F)./TumSD);

Fout = SQ_Res^2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function Fout = modeleval(lam, k, inttum, Time)

% options = odeset('RelTol',1e-5,'AbsTol',1e-7);


Fout = inttum*k./(inttum + (k-inttum)*exp(-lam*(Time-Time(1))));

function Fout = SofNeval(inttum, n, params)

rho = params(1);
gam = params(2);

Fout = gam./(1+(gam-1)*exp(rho*(n-inttum)));

function Fout = weibulleval(time,p)

alpha = p(1);
beta = p(2);
eta = p(3);

Fout = exp(-(((time+alpha)/eta).^beta));

% Function to determine the significance level
function sig = get_significance(p)
    if p < 0.001
        sig = '***';
    elseif p < 0.01
        sig = '**';
    elseif p < 0.05
        sig = '*';
    else
        sig = 'ns';
    end

function plot_iqr_box(x_position, data)
    q1 = quantile(data, 0.25); % First quartile (25%)
    q3 = quantile(data, 0.75); % Third quartile (75%)
    median_value = median(data); % Median

    % Plot the box for IQR
    rectangle('Position', [x_position-0.1, q1, 0.2, q3-q1], 'EdgeColor', 'k', 'LineWidth', 1.5);

    % Plot the median line
    plot([x_position-0.1, x_position+0.1], [median_value, median_value], 'k-', 'LineWidth', 1.5);








