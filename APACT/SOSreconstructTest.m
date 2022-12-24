clear all;
close all;

Step          = 0.00004;
WholeImageX   = -0.015: Step: 0.015;
WholeImageY   = -0.015: Step: 0.015;

HalfWindow    = 0.0016;
ImageX        = - HalfWindow:Step:HalfWindow;
ImageY        = - HalfWindow:Step:HalfWindow;


R_media = 1e-2;
R_view  = 1.4e-2;
R_ring  = 0.05;

V_media = 1600;
V_water = 1500;
V_sound = 1520;

FFT_half_N         = 128;
f = ((1:FFT_half_N * 2) - FFT_half_N - 1)/(2 * FFT_half_N) * 1/Step * 2 * pi;
[AngularFrequency_x, AngularFrequency_y] = meshgrid(f);

AngularFrequency = sqrt(AngularFrequency_x.^2 + AngularFrequency_y.^2);

Theta = acos(AngularFrequency_x ./ AngularFrequency);
Theta(AngularFrequency_y < 0) = -Theta(AngularFrequency_y < 0);


[WholeImageXX, WholeImageYY] = meshgrid(WholeImageX, WholeImageY);
Mask = WholeImageXX.^2 + WholeImageYY.^2 < R_view^2;

[Look_x, Look_y, Look_x_id, Look_y_id] =...
    FormPatchCenters(WholeImageX, WholeImageY, Mask, ImageX, ImageY);

Wavefront_a      = cell(length(Look_x), 1);

parfor k = 1:length(Look_x)
    k
    Wavefront_a{k} = WavefrontAnalytical(Look_x(k), Look_y(k), R_media, R_ring, V_water, V_media, V_sound, Theta, FFT_half_N);
end

ShowWavefrontsMap(Look_x, Look_y, Wavefront_a, FFT_half_N, ones(length(Look_x), 1), [1,1,0]);

%%
clear all;
load('G:\光声成像\局部傅里叶变换自适应重建算法\BigData\IN_VIVO_RESULT.mat');
%load('G:\光声成像\局部傅里叶变换自适应重建算法\BigData\SIMULATION_RESULT.mat');
SOS = reconstructSOS(WholeImageX(1:5:end), WholeImageY(1:5:end), Look_x, Look_y, Wavefront, FFT_half_N, Res, V_sound, R_ring);

figure;
imagesc(WholeImageX(1:5:end), WholeImageY(1:5:end), SOS);
axis square;
set(gca,'YDir','normal');
axis off;



