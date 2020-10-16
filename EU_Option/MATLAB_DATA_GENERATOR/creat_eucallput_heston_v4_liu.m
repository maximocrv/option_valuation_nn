clear
% this function is to generate the training samples using Latin hypercube
% version 3 @shuaiqiang
% sampling for the Heston model
rng(67)

n=100000; % number of the samples
%Smax=1.6;      Smin=0.4;
Tmin=0.5;     Tmax=1.4;


%---Heston parameters----
rhomin=-0.8;      rhomax=-0.3;
kapmin=0.1;     kapmax=1.0;
gammamin=0.01;   gammamax=0.5; % sig=gamma
vbarmin = 0.01;  vbarmax=0.5; % th=vbar
rmin=0.0;          rmax=0.1;
v0min=0.01;      v0max=0.5;
Kmin=0.6;          Kmax=1.4;

S     = 1.0+zeros(n,1);
lowerbound = [Kmin Tmin rmin  rhomin kapmin gammamin vbarmin v0min];
upperbound = [Kmax Tmax rmax  rhomax kapmax gammamax vbarmax v0max ];
 [X_scaled,X_normalized]=lhsdesign_modified(n,lowerbound,upperbound);
 K     = X_scaled(:,1);
 T   = X_scaled(:,2);
 r   = X_scaled(:,3);
 rho = X_scaled(:,4);
 kap = X_scaled(:,5);
 gamma = X_scaled(:,6);
 vbar  = X_scaled(:,7);
 v0    = X_scaled(:,8);



% % feller condition 
% feller = 2.0*kap.*th-sig;

%--put option---

%---the final setting---
LL = 12;
NN = LL*30;

%%% CALCULATING HESTON CALL AND PUT VALUES %%%
tic
Vp_H =zeros(size(S));
for i=1:size(Vp_H)
    Vp_H(i) = HSTeuPut_COS_LN(S(i),K(i),T(i),r(i),v0(i),kap(i),vbar(i),gamma(i),rho(i), LL, NN);
end
toc

%put-call parity. check whether the put value is between upper and lower
valid_index = (Vp_H < K.*exp(-r.*T)) & (Vp_H>K.*exp(-r.*T)-S);

%valid_index = Vp>0;
Vc_H =Vp_H+S-K.*exp(-r.*T);

%-- bound check--
Vp_H = Vp_H(valid_index);
Vc_H = Vc_H(valid_index);
T = T(valid_index);
K = K(valid_index);
v0 = v0(valid_index);
S = S(valid_index); 
rho = rho(valid_index);
kap = kap(valid_index);
gamma = gamma(valid_index);
vbar = vbar(valid_index);
r = r(valid_index);


%%% CALCULATING HESTON NUMERICAL DERIVATIVES %%%%

h=1e-5;

dPdS_H = zeros(size(Vp_H));
dCdS_H = zeros(size(Vc_H));
for i=1:size(Vp_H)
    Vp_plus = HSTeuPut_COS_LN(S(i) + h, K(i), T(i), r(i), v0(i), kap(i), vbar(i), gamma(i), rho(i), LL, NN);
    Vp_minus = HSTeuPut_COS_LN(S(i) - h, K(i), T(i), r(i), v0(i), kap(i), vbar(i), gamma(i), rho(i), LL, NN);
    Vc_plus = Vp_plus + S(i) + h - K(i) * exp(-r(i) * T(i));
    Vc_minus = Vp_minus + S(i) - h - K(i) * exp(-r(i) * T(i));
    dPdS_H(i) = (Vp_plus - Vp_minus)/(2*h);
    dCdS_H(i) = (Vc_plus - Vc_minus)/(2*h);
end



dPdK_H = zeros(size(Vp_H));
dCdK_H = zeros(size(Vc_H));
for i=1:size(Vp_H)
    Vp_plus = HSTeuPut_COS_LN(S(i), K(i) + h, T(i), r(i), v0(i), kap(i), vbar(i), gamma(i), rho(i), LL, NN);
    Vp_minus = HSTeuPut_COS_LN(S(i), K(i) - h, T(i), r(i), v0(i), kap(i), vbar(i), gamma(i), rho(i), LL, NN);
    Vc_plus = Vp_plus + S(i) - (K(i) + h) * exp(-r(i) * T(i));
    Vc_minus = Vp_minus + S(i) - (K(i) - h) * exp(-r(i) * T(i));
    dPdK_H(i) = (Vp_plus - Vp_minus)/(2*h);
    dCdK_H(i) = (Vc_plus - Vc_minus)/(2*h);
end


dPdT_H = zeros(size(Vp_H));
dCdT_H = zeros(size(Vc_H));
for i=1:size(Vp_H)
    Vp_plus = HSTeuPut_COS_LN(S(i), K(i), T(i) + h, r(i), v0(i), kap(i), vbar(i), gamma(i), rho(i), LL, NN);
    Vp_minus = HSTeuPut_COS_LN(S(i), K(i), T(i) - h , r(i), v0(i), kap(i), vbar(i), gamma(i), rho(i), LL, NN);
    Vc_plus = Vp_plus + S(i) - K(i) * exp(-r(i) * (T(i) + h));
    Vc_minus = Vp_minus + S(i) - K(i) * exp(-r(i) * (T(i) - h));
    dPdT_H(i) = (Vp_plus - Vp_minus)/(2*h);
    dCdT_H(i) = (Vc_plus - Vc_minus)/(2*h);
end


%%% CALCULATING ANALYTIC BS DERIVATIVES %%%
mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);

dCdK_BS = zeros(size(Vp_H));
dPdK_BS = zeros(size(Vp_H));
dCdT_BS = zeros(size(Vp_H));
dPdT_BS = zeros(size(Vp_H));
for i=1:size(Vp_H)
    T_sqrt = sqrt(T(i));
    d1 = (log(S(i)/K(i))+(r(i)+vbar(i)*vbar(i)/2.)*T(i))/(vbar(i)*T_sqrt);
    d2 = d1-vbar(i)*T_sqrt;
    dCdK_BS(i) = -exp(-r(i)*T(i))*cdf(pd, d2);
    dPdK_BS(i) = exp(-r(i)*T(i))*cdf(pd, -d2);
    dCdT_BS(i) = -1*(-(S(i)*vbar(i)*pdf(pd, d1))/(2*T_sqrt)-r(i)*K(i)*exp(-r(i)*T(i))*cdf(pd, d2));  %*-1 due to dV/dt = -dV/dT!! Q10.5 Higham
    dPdT_BS(i) = dCdT_BS(i) - K(i)*r(i)*exp(-r(i)*T(i));
end



uT_index = dCdT_H > 0;
Vp_H = Vp_H(uT_index);
Vc_H = Vc_H(uT_index);
T = T(uT_index);
K = K(uT_index);
v0 = v0(uT_index);
S = S(uT_index);
rho = rho(uT_index);
kap = kap(uT_index);
gamma = gamma(uT_index);
vbar = vbar(uT_index);
r = r(uT_index);

dCdS_H = dCdS_H(uT_index);
dCdT_H = dCdT_H(uT_index);
dCdK_H = dCdK_H(uT_index);
dPdS_H = dPdS_H(uT_index);
dPdT_H = dPdT_H(uT_index);
dPdK_H = dPdK_H(uT_index);

dCdT_BS = dCdT_BS(uT_index);
dCdK_BS = dCdK_BS(uT_index);
dPdT_BS = dPdT_BS(uT_index);
dPdK_BS = dPdK_BS(uT_index);





%39,46 seeds
%Using blsprice and calcGreeks toolboxes
[Vc_BS, Vp_BS] = blsprice(S, K, r, T, vbar);
BS_input_data = [S T K r vbar];
BS_output_data = [Vc_BS, Vp_BS];
BS_analytic_grads = [dCdT_BS, dPdT_BS, dCdK_BS,  dPdK_BS];

heston_input_data = [S T K r rho kap gamma v0 vbar];
heston_output_data = [Vc_H Vp_H];
heston_numerical_grads = [dCdS_H, dPdS_H, dCdT_H, dPdT_H, dCdK_H, dPdK_H];

csvwrite('/Users/Maximocravero/PycharmProjects/Finance Research/EU_Option/BS_Constraint/data/BS_input_data.csv', BS_input_data)
csvwrite('/Users/Maximocravero/PycharmProjects/Finance Research/EU_Option/BS_Constraint/data/BS_output_data.csv', BS_output_data)
csvwrite('/Users/Maximocravero/PycharmProjects/Finance Research/EU_Option/BS_Constraint/data/BS_analytic_grads.csv', BS_analytic_grads)

csvwrite('/Users/Maximocravero/PycharmProjects/Finance Research/EU_Option/Heston_Constraint/data/Heston_input_data.csv', heston_input_data)
csvwrite('/Users/Maximocravero/PycharmProjects/Finance Research/EU_Option/Heston_Constraint/data/Heston_output_data.csv', heston_output_data)
csvwrite('/Users/Maximocravero/PycharmProjects/Finance Research/EU_Option/Heston_Constraint/data/Heston_numerical_grads.csv', heston_numerical_grads)
