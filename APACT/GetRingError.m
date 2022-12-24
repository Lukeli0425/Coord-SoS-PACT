
R_ring        = 0.05;
T_sample      = 1/40e6;


% load('G:\光声成像\局部傅里叶变换自适应重建算法\RYS_20200612.mat');
% Delay_id = 46;
% Sinogram = Sino(Delay_id : end, :);
% V_sound = 1500;

load('G:\光声成像\局部傅里叶变换自适应重建算法\KEXIN_HAIR.mat')
Delay_id = 48;
Sinogram = Sinogram1(Delay_id : end, :);
V_sound = 1499;

load('EIR_TUNGSTEN.mat');
ht = EIR_AVE_20180516_LEFT(Delay_id : end); 
t0 = 1120 - Delay_id + 1; 

Sinogram  = PhaseCorrectPASignal(Sinogram, ht, t0);

N_transducer = size(Sinogram, 2);
N_time       = size(Sinogram, 1);

Step = 0.00004;

WholeImageX = -0.015: Step: 0.015;
WholeImageY = -0.015: Step: 0.015;



WholeImage = DelayAndSumReconstruction...
    (R_ring, T_sample, V_sound, Sinogram, WholeImageX, WholeImageY); 

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

[x, y] = ginput(1);

[~, TOFid] = max(Sinogram);

TOF = TOFid * T_sample;

delta_angle = 2*pi/N_transducer;
transducer_theta = delta_angle * (1:N_transducer);

x_transducer = R_ring * cos(transducer_theta);
y_transducer = R_ring * sin(transducer_theta);

d = sqrt((x - x_transducer).^2 + (y - y_transducer).^2);

e = TOF * V_sound - d;

figure;
plot(e);

figure(100);
hold on;
not_broken = e > -0.01 & e < 0.01;
tr = 1:length(e);
e = interp1(tr(not_broken), e(not_broken), tr);
plot(e);







