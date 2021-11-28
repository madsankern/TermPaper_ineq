% Code to solve the consumption-saving model

clear all; clc; close all;

%% Model setup

% Define parameters
ga = 2; % Intertemporal substitution
rho = 0.05; % Discount rate
z1 = .01; % Low productivity
z2 = 3*z1; % High productivity (originally at 3*z1)
z = [z1,z2];
la1 = 0.5; % lambda_1
la2 = 0.5; % lambda_2
la = [la1,la2];

w = 3; % Baseline wage
R = 0.051; % Averate risky return
r = 0.041; % Riskless return
phi = 0.3; % Borrowing constraint for riskless asset

zeta = 1.5; % Asymptotic tail exponent
sig2 = (zeta/ga + 1)/2*(R-r)^2/(rho - r); % Set risky variance (sig2) so that zeta=1.5
sig = sig2^(1/2);

I= 20000; % Points in the asset grid
amin = -phi; % Minimum assets
amax = 3000; % Maximum assets

% Define a non-uniform grid for assets
x = linspace(0,1,I)';
coeff = 5; power = 10;
xx  = x + coeff*x.^power;
xmax = max(xx); xmin = min(xx);
a = (amax-amin)/(xmax - xmin)*xx + amin;
daf = ones(I,1);
dab = ones(I,1);
daf(1:I-1) = a(2:I)-a(1:I-1);
dab(2:I) = a(2:I)-a(1:I-1);
daf(I)=daf(I-1); dab(1)=dab(2);

aa = [a,a];
daaf = daf*ones(1,2);
daab = dab*ones(1,2);

%objects for approximation of second derivatives
denom = 0.5*(daaf + daab).*daab.*daaf;
weightf = daab./denom;
weight0 = -(daab + daaf)./denom;
weightb = daaf./denom;

zz = ones(I,1)*z;
kk = ones(I,2); % Check this

maxit = 30; % Maximum iterations
crit = 10^(-6); % Tolerance for convergence
Delta = 1000; % Stepsize for implicit method

% Initialize forward/backwards diffierence for value function and
% consumption
dVf = zeros(I,2);
dVb = zeros(I,2);
dV0 = zeros(I,2);
dV2 = zeros(I,2);
c = zeros(I,2);
c0 = zeros(I,2);

% Helper for the transition matrix
Aswitch = [-speye(I)*la(1),speye(I)*la(1);speye(I)*la(2),-speye(I)*la(2)];

%INITIAL GUESS
v0(:,1) = (w*z(1) + r*a).^(1-ga)/(1-ga)/rho;
v0(:,2) = (w*z(2) + r*a).^(1-ga)/(1-ga)/rho;

% v0(:,1) = (w*z(1) + r*a + (R-r)^2/(s*sig2)*a).^(1-s)/(1-s)/rho;
% v0(:,2) = (w*z(2) + r*a + (R-r)^2/(s*sig2)*a).^(1-s)/(1-s)/rho;

v = v0;
%% Run the finite difference algorithm

tic;
for n=1:maxit

    disp(n)
    
    V = v; 
    V_n(:,:,n)=V;
    
    % Compute forward difference
    dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))./(aa(2:I,:) - aa(1:I-1,:));
    dVf(I,:) = (w*z + r.*amax + (R-r)^2/(ga*sig2)*amax).^(-ga); %will never be used
    
    % Compute backward difference
    dVb(2:I,:) = (V(2:I,:)-V(1:I-1,:))./(aa(2:I,:) - aa(1:I-1,:));
    dVb(1,:) = (w*z + r.*amin).^(-ga); %state constraint boundary condition

    % Second derivative: approximation only differs at amax
    dV2b(2:I-1,:) = (daab(2:I-1,:).*V(3:I,:) - (daab(2:I-1,:)+daaf(2:I-1,:)).*V(2:I-1,:) + daaf(2:I-1,:).*V(1:I-2,:))./denom(2:I-1,:);
    dV2f(2:I-1,:) = (daab(2:I-1,:).*V(3:I,:) - (daab(2:I-1,:)+daaf(2:I-1,:)).*V(2:I-1,:) + daaf(2:I-1,:).*V(1:I-2,:))./denom(2:I-1,:);
    dV2b(I,:) = -ga*dVb(I,:)/amax;
    dV2f(I,:) = -ga*dVf(I,:)/amax;
     
    % Consumption and savings with forward difference
    cf = max(dVf,10^(-10)).^(-1/ga);
    kf = max(- dVf./dV2f.*(R-r)/sig2,0);
    kf = min(kf,aa+phi);
    ssf = w*zz + (R-r).*kf + r.*aa - cf;
    
    % Consumption and savings with backward difference
    cb = max(dVb,10^(-10)).^(-1/ga);
    kb = max(- dVb./dV2b.*(R-r)/sig2,0);
    kb = min(kb,aa+phi);
    ssb = w*zz + (R-r).*kb + r.*aa - cb;
    
    % Consumption and derivative of value function at steady state
    k0 = (kb + kf)/2;
    c0 = w*zz + (R-r).*k0 + r.*aa;
    dV0 = max(c0,10^(-10)).^(-ga);
   
    % dV_upwind makes a choice of forward or backward differences based on the sign of the drift    
    If = real(ssf) > 10^(-12); %positive drift --> forward difference
    Ib = (real(ssb) < -10^(-12)).*(1-If); %negative drift --> backward difference
    I0 = (1-If-Ib); %at steady state
   
    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0; % Define dV from the upwind scheme
    c = max(dV_Upwind,10^(-10)).^(-1/ga); % Consumption from the derivative of the value function
    u = c.^(1-ga)/(1-ga); % Utility of choice
    k = max(-dV_Upwind./dV2b.*(R-r)/sig2,0); % Investment in risky asset as 
    k = min(k,aa+phi);
    
    % Construct transition matrix
    X = -Ib.*ssb./daab + sig2/2.*k.^2.*weightb;
    Y = - If.*ssf./daaf + Ib.*ssb./daab + sig2/2.*k.^2.*weight0;
    Z = If.*ssf./daaf + sig2/2.*k.^2.*weightf; 
    
    xi = -amax*(R-r)^2/(2*ga*sig2);
    X(I,:) = -min(ssb(I,:),0)./daab(I,:) - xi./daab(I,:);
    Y(I,:) = -max(ssf(I,:),0)./daaf(I,:) + min(ssb(I,:),0)./daab(I,:) + xi./daab(I,:);
    Z(I,:) = max(ssf(I,:),0)./daaf(I,:);
    
    A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
    A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);
    A1(I,I) = Y(I,1) + Z(I,1);
    A2(I,I) = Y(I,2) + Z(I,2);
    A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;
    
   if max(abs(sum(A,2)))>10^(-9)
       disp('Improper Transition Matrix')
       break
   end
    
    % Construct new transition matrix
    B = (1/Delta + rho)*speye(2*I) - A;
    
    % Stack solutions
    u_stacked = [u(:,1);u(:,2)];
    V_stacked = [V(:,1);V(:,2)];
    
    b = u_stacked + V_stacked/Delta;
    V_stacked = B\b; % Solve the matrix equation by inverting B
    
    V = [V_stacked(1:I),V_stacked(I+1:2*I)];
    
    % Compute change in the value function
    Vchange = V - v;
    v = V;
    
    % Check for convergence
    dist(n) = max(max(abs(Vchange)));
    if dist(n)<crit
        disp('Value Function Converged, Iteration = ')
        disp(n)
        break
    end
end
toc;

%% Solve the Kolmogorov forward equation

%RECOMPUTE TRANSITION MATRIX WITH REFLECTING BARRIER AT amax
X = -min(ssb,0)./daab + sig2/2.*k.^2.*weightb;
Y = -max(ssf,0)./daaf + min(ssb,0)./daab + sig2/2.*k.^2.*weight0;
Z = max(ssf,0)./daaf + sig2/2.*k.^2.*weightf; 

A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);
A1(I,I) = Y(I,1) + Z(I,1);
A2(I,I) = Y(I,2) + Z(I,2);
A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;

%WORK WITH RESCALED DENSITY \tilde{g} BELOW
da_tilde = 0.5*(dab + daf);
da_tilde(1) = 0.5*daf(1); da_tilde(I) = 0.5*dab(I);
da_stacked = [da_tilde;da_tilde];
grid_diag = spdiags(da_stacked,0,2*I,2*I);

AT = A';
b = zeros(2*I,1);

%need to fix one value, otherwise matrix is singular
i_fix = 1;
b(i_fix)=.1;
row = [zeros(1,i_fix-1),1,zeros(1,2*I-i_fix)];
AT(i_fix,:) = row;

%Solve linear system
g_tilde = AT\b;

%rescale \tilde{g} so that it sums to 1
g_sum = g_tilde'*ones(2*I,1);
g_tilde = g_tilde./g_sum;

gg = grid_diag\g_tilde; %convert from \tilde{g} to g

g = [gg(1:I),gg(I+1:2*I)];

check1 = g(:,1)'*da_tilde;
check2 = g(:,2)'*da_tilde;
%%

%CALCULATE THEORETICAL POWER LAW EXPONENT
zeta = ga*(2*sig2*(rho - r)/(R-r)^2 -1);

adot = w*zz + (R-r)*k + r.*aa - c;

risky_share = (R-r)/(ga*sig2);
cslope = (rho - (1-ga)*r)/ga - (1-ga)/(2*ga)*(R-r)^2/(ga*sig2);
sbar = (r-rho)/ga + (1+ga)/(2*ga)*(R-r)^2/(ga*sig2);

plot(a,k,a,risky_share.*a)
plot(a,c,a,cslope.*a)

%COMPUTE DISTRIBUTION OF x=log(a)
for i=1:I
    G(i,1) = sum(g(1:i,1).*da_tilde(1:i));
    G(i,2) = sum(g(1:i,2).*da_tilde(1:i));
end

f = zeros(I,2);
x = log(max(a,0));
dx = zeros(I,1);
for i =2:I
dx(i) = x(i)-x(i-1);
f(i,1) = (G(i,1)-G(i-1,1))/dx(i);
f(i,2) = (G(i,2)-G(i-1,2))/dx(i);
end
f(1)=0;

xmin = log(1); xmax = log(amax);
ga = g(1:i,1) + g(1:i,2);
fx = f(:,1)+f(:,2);

%% Plot to check results
%g_1 = g_tilde(1:I);
%g_2 = g_tilde(I+1:2*I);
%g_tot = (g_1 + g_2)/2;

g_tot = (g(:,1) + g(:,2))/2;

plot(a,g_tot)
xlim([amin 10])

%%
plot(a,adot)
xlim([amin 10])

%% Save wealth distribution as a .cvs file
writematrix(g_tot, 'ga_var.csv');
writematrix(a, 'a_var.csv'); 
writematrix(adot, 'saving.csv');
writematrix(k, 'risky.csv');

