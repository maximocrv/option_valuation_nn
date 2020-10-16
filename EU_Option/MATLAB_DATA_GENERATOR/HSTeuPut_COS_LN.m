 %Copyright (C) 2015 M.J. Ruijter

    %This file is part of BENCHOP.
    %BENCHOP is free software: you can redistribute it and/or modify
    %it under the terms of the GNU General Public License as published by
    %the Free Software Foundation, either version 3 of the License, or
    %(at your option) any later version.

    %BENCHOP is distributed in the hope that it will be useful,
    %but WITHOUT ANY WARRANTY; without even the implied warranty of
    %MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    %GNU General Public License for more details.

    %You should have received a copy of the GNU General Public License
    %along with BENCHOP. If not, see <http://www.gnu.org/licenses/>.
function [U] = HSTeuPut_COS_LN(S,K,T,r,V,kap,th,sig,rho,L,N)
% BENCHOP Problem 4: The Heston model for one underlying asset
% HSTeuCall_COS computes the price for a European call option
%
% Input:    S       - Initial asset price   
%           K       - Strike price
%           T       - Terminal time  
%           r       - Risk-free interest rate
%           V       - Initial instantaneous variance
%           kap     - Speed of mean reversion
%           th      - Long run variance
%           sig     - Volatility of variance process
%           rho     - Correlation coefficient
%
% Output:   U       - Option value
%
% This MATLAB code has been written for the BENCHOP project and is based on 
% the COS methodes developed by F. Fang, C.W. Oosterlee, and M.J. Ruijter

% Parameters
x = log(S./K);

% Cumulants
c1 = r*T+(1-exp(-kap*T))*(th-V)/(2*kap)-0.5*th*T;
c2alt = th*(1+sig)*T;
% Interval [a,b]
%shuaiqiangliu: here seems an error, x is missing?? otherwise, deep in the
%money option can not have a proper vaule.
%L = 12; %

a = c1-L*abs(sqrt(c2alt))+x;
b = c1+L*abs(sqrt(c2alt))+x;


% Number of Fourier cosine coefficients
%N = 512;
k = 0:N-1;


omega = k'*pi/(b-a);

% Fourier cosine coefficients payoff function
omegamina = omega*(-a);
%omegabmina = omega*(b-a);
% put option payoff
cp=-1;
chi =  (cos(omegamina)+omega.*sin(omegamina)-exp(a) )./(1+omega.^2);
psi = sin(omegamina)./omega;
psi(1) = -a;
Uk = cp*(chi-psi)';
Uk(1) = 0.5*Uk(1);


% Characteristic function
D = sqrt((kap-1i*rho*sig*omega).^2+(omega.^2+1i*omega)*sig^2);
G = (kap-1i*rho*sig*omega-D)./(kap-1i*rho*sig*omega+D);
cf = exp(1i*omega*r*T+V/sig^2*(1-exp(-D*T))./(1-G.*exp(-D*T)) ...
     .*(kap-1i*rho*sig*omega-D))...
     .*exp(kap*th/sig^2*(T*(kap-1i*rho*sig*omega-D)-2*log((1-G.*exp(-D*T))./(1-G))));

% Fourier cosine coefficients density function
Recf = real(repmat(cf,1,length(x)).*exp(1i*omega*(x-a)));

% Option value 
U =double( exp(-r*T)*2/(b-a)*K*(Uk*Recf));

end