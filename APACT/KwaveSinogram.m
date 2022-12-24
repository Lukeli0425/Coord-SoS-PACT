close all;

N_transducer = 512;
R_ring       = 0.05;
T_sample     = 1/40e6;
N_time       = 2000;

I = imread('P0_3.bmp');
P0 = double(I(:, :, 1));
P0 = (255 - P0)/255;
% P0(P0 == 0)   = 1;
% P0(P0 == 255) = 0;

[Ny, Nx] = size(P0);
dy = 1.1 * 2 * R_ring / Ny;
dx = 1.1 * 2 * R_ring / Nx;

X = dx * ((1:Nx) - round(Nx / 2));
Y = dy * ((1:Ny) - round(Ny / 2));

% P0 = 0 * P0;
% P0(800, 800) = 1;
% point_x = X(800);
% point_y = Y(800);

[XX, YY] = meshgrid(X, Y);
R = 1e-2;
R1 = 6e-3;
offset = 2e-3;
SOS = ones(Ny, Nx) * 1500;
SOS(XX.^2 + YY.^2 < R^2) = 1600;
SOS(XX.^2 + (YY + offset).^2 < R1^2) = 1650;

%%

Sinogram = kWaveExperiment(P0, SOS, dx, dy, Nx, Ny, 1500, R_ring, N_transducer, T_sample, N_time);

figure;
imagesc(Sinogram);
colormap gray; 

%%
Step  = 0.00004;
V_sound = 1520;
WholeImageX = -0.015: Step: 0.015;
WholeImageY = -0.015: Step: 0.015;

WholeImage = DelayAndSumReconstruction...
    (R_ring, T_sample, V_sound, Sinogram, WholeImageX, WholeImageY); 

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
axis square;
colormap gray;
set(gca,'YDir','normal');
caxis([min(min(WholeImage)), max(max(WholeImage))]);
axis off;

%%

% figure;
% imagesc(X, Y, flipud(P0'));
% colormap gray;
% axis square;
% hold on;
% theta = (1:101) / 100 * 2 * pi;
% plot(R * cos(theta), R * sin(theta));
% drawnow;
% 
% hold on;
% bar_x = [10*Step, 35*Step] -1e-2;
% bar_y = -[10*Step, 10*Step] + 1e-2;
% plot(bar_x, bar_y, 'LineWidth', 3, 'Color', [1,1,1]);
% 
% 
% figure;
% imagesc(X, Y, SOS');
% axis square;
% drawnow;


%%
[WholeImageXX, WholeImageYY] = meshgrid(WholeImageX, WholeImageY);
Mask = WholeImageXX.^2 + WholeImageYY.^2 < R^2;

HalfWindow = 0.0016;

ImageX = - HalfWindow:Step:HalfWindow;
ImageY = - HalfWindow:Step:HalfWindow;

FFT_half_N = 128;

[Look_x, Look_y, Look_x_id, Look_y_id] =...
FormPatchCenters(WholeImageX, WholeImageY, Mask, ImageX, ImageY);

wavefront = cell(length(Look_x), 1);
TOF       = cell(length(Look_x), 1);
for k = 1:length(Look_x)
    k
    [TOF{k}, wavefront{k}] =...
    WavefrontAnalytical2(Look_x(k), Look_y(k), N_transducer, ...
                        R, R1, [0,0], [- offset, 0], R_ring,...
                        1500, 1600, 1650, V_sound, theta);
    figure(11);
    plot(wavefront{k});
end

%%
fs = (1:2 * FFT_half_N) - FFT_half_N;
fs = fs / FFT_half_N;
fs = fs * HalfWindow / 4 * 0.9;

f = ((1:FFT_half_N * 2) - FFT_half_N - 1)/(2 * FFT_half_N) * 1/Step * 2 * pi;
[AngularFrequency_x, AngularFrequency_y] = meshgrid(f);

AngularFrequency = sqrt(AngularFrequency_x.^2 + AngularFrequency_y.^2);

Theta = acos(AngularFrequency_x ./ AngularFrequency);
Theta(AngularFrequency_y < 0) = -Theta(AngularFrequency_y < 0);

for k = 1:length(Look_x)

    Wavefront = WavefrontAnalytical(Look_x(k), Look_y(k), R, R_ring,...
                1500, 1600, 1520, Theta, FFT_half_N);

    figure(1998)
    hold on;
    imagesc(Look_x(k) + fs, Look_y(k) + fs, Wavefront);
    
    xlim([-0.015, 0.015]);
    ylim([-0.015, 0.015]);
    drawnow;
end


