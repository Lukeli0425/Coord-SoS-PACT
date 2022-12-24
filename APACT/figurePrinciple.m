Step               = 0.00004;
DelayDistance_step = 0.00001;
HalfWindow         = 0.0016;
HalfDelay          = 0.0008;

V_sound            = 1520;
FFT_half_N         = 128;

R_ring        = 0.05;
T_sample      = 1/40e6;

ImageX        = - HalfWindow:Step:HalfWindow;
ImageY        = - HalfWindow:Step:HalfWindow;
DelayDistance = - HalfDelay:DelayDistance_step:HalfDelay;

load('EIR_KWAVE.mat');
t0 = 1520;

N_transducer = 512;
delta_angle  = 2*pi/N_transducer;
angle_transducer = delta_angle * (1:N_transducer);
wavefront_function = 3e-5 * cos(angle_transducer) -...
                     3e-5 * sin(angle_transducer) +...
                     1e-4 * cos(2 * angle_transducer);

TOF = (R_ring - wavefront_function) / V_sound;

Sinogram = zeros(length(ht), N_transducer);
for k = 1:N_transducer
    Sinogram(:, k) = circshift(ht, round(TOF(k) / T_sample - t0));
end

NewSinogram  = PhaseCorrectPASignal(Sinogram, ht, t0);
Sinogram     = NewSinogram;

Sinogram0 = zeros(length(ht), N_transducer);
for k = 1:N_transducer
    Sinogram0(:, k) = circshift(ht, round(R_ring / V_sound / T_sample - t0));
end

NewSinogram  = PhaseCorrectPASignal(Sinogram0, ht, t0);
Sinogram0     = NewSinogram;


%%
PSF = DelayAndSumReconstruction...
    (R_ring, T_sample, V_sound, Sinogram, ImageX, ImageY); 
PSF0 = DelayAndSumReconstruction...
    (R_ring, T_sample, V_sound, Sinogram0, ImageX, ImageY); 

I = imread('FIGURE_FEATURE.bmp');
P0 = double(I(:, :, 1));
P0 = (255 - P0) / 255;

Image = conv2(PSF, P0, 'same');
Image0 = conv2(PSF0, P0, 'same');

figure;
imagesc(ImageX, ImageY, Image);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

figure;
imagesc(ImageX, ImageY, PSF);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

figure;
imagesc(ImageX, ImageY, Image0);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

figure;
imagesc(ImageX, ImageY, PSF0);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

%%
f = ((1:FFT_half_N * 2) - FFT_half_N - 1)/(2 * FFT_half_N) * 1/Step * 2 * pi;
[AngularFrequency_x, AngularFrequency_y] = meshgrid(f);

AngularFrequency = sqrt(AngularFrequency_x.^2 + AngularFrequency_y.^2);

Theta = acos(AngularFrequency_x ./ AngularFrequency);
Theta(AngularFrequency_y < 0) = - Theta(AngularFrequency_y < 0);

Wavefront = 3e-5 * cos(Theta) -...
            3e-5 * sin(Theta) +...
            1e-4 * cos(2 * Theta);


Wavefront_backward = rot90(Wavefront, 2);
Wavefront_backward = circshift(Wavefront_backward, 1, 1);
Wavefront_backward = circshift(Wavefront_backward, 1, 2);

T   = zeros([size(Wavefront), length(DelayDistance)]);

for t = 1:length(DelayDistance)
    A_forward  = exp(-1i * AngularFrequency.* (- Wavefront + DelayDistance(t)));
    A_backward = exp( 1i * AngularFrequency.* (- Wavefront_backward + DelayDistance(t)));
    T(:, :, t) = (A_forward + A_backward) / 2;
    T(FFT_half_N + 1, FFT_half_N + 1, t) = 1;
end

%%

[PSFVolume, FPSFVolume] = FastGenerate3DVolume...
(Sinogram, R_ring, T_sample, V_sound, ImageX, ImageY, DelayDistance, FFT_half_N);


figure;
s = slice(ImageX, ImageY, DelayDistance, ...
          PSFVolume, [], [], DelayDistance(1:20:end));
colormap gray;
set(s,'EdgeColor','none');

zlabel('d');

set(gcf,'Units','centimeter','Position',[1 1 16 20])
view([37.5,10]);

axis off;
%%

ImageVolume = zeros(size(PSFVolume));
FTVolume    = zeros(2 * FFT_half_N, 2 * FFT_half_N, length(DelayDistance));

for r = 1:length(DelayDistance)
    ImageVolume(:, :, r) = conv2(PSFVolume(:, :, r), P0, 'same');
end

GaussWindow = gausswin(length(ImageY)) * (gausswin(length(ImageX)))';

for r = 1:length(DelayDistance)
    FTVolume(:, :, r) = Myfft(ImageVolume(:, :, r).* GaussWindow, FFT_half_N).* AngularFrequency;
end

%%
figure;
s = slice(ImageX, ImageY, DelayDistance, ...
          ImageVolume, [], [], DelayDistance(1:20:end));
colormap gray;
set(s,'EdgeColor','none');

zlabel('d');  

set(gcf,'Units','centimeter','Position', [1 1 16 20])
view([37.5,10]);

axis off;

%%
theta = (1:512) / 512 * 2 * pi;
fr = (0:60) /(2 * FFT_half_N) * 1/Step * 2 * pi;

Fxq = fr' * cos(theta);
Fyq = fr' * sin(theta);

PolarFT = zeros(length(fr), length(theta), length(DelayDistance));
PolarTF = zeros(length(fr), length(theta), length(DelayDistance));

for d = 1:length(DelayDistance)
    PolarFT(:, :, d) = interp2(f, f, abs(FTVolume(:, :, d)), Fxq, Fyq);
    PolarTF(:, :, d) = interp2(f, f, abs(T(:, :, d)), Fxq, Fyq);
end

figure;
hold on;
s = slice(f, f, DelayDistance, ...
          abs(FTVolume), [], [], DelayDistance(1:20:end));
colormap gray;
set(s,'EdgeColor','none');

set(s,'facealpha',0.3);

Z = DelayDistance' * ones(1, length(fr));
for k = 1:32:384
    X = ones(length(DelayDistance), 1) * fr * cos(theta(k));
    Y = ones(length(DelayDistance), 1) * fr * sin(theta(k));
    s = surf(X, Y, Z, squeeze(PolarFT(:, k, :))', 'EdgeColor', 'none');
    set(s,'facealpha',1);
    
    colormap gray;
end

% Z = DelayDistance' * ones(1, length(theta));
% X = ones(length(DelayDistance), 1) * fr(40) * cos(theta);
% Y = ones(length(DelayDistance), 1) * fr(40) * sin(theta);
% s = surf(X, Y, Z, squeeze(PolarFT(40, :, :))', 'EdgeColor', 'none');
% set(s,'facealpha',0.8);


set(gcf,'Units','centimeter','Position', [1 1 16 20])
view([37.5,10]);
% caxis([0, 1e15]);

xlim([-max(fr), max(fr)]);
ylim([-max(fr), max(fr)]);
axis off;

%%
figure;
hold on;
% s = slice(f, f, DelayDistance, ...
%           abs(T), [], [], DelayDistance(1:20:end));
% colormap gray;
% set(s,'EdgeColor','none');
% 
% set(s,'facealpha',0.1);

Z = DelayDistance' * ones(1, length(fr));
for k = 1:32:384
    X = ones(length(DelayDistance), 1) * fr * cos(theta(k));
    Y = ones(length(DelayDistance), 1) * fr * sin(theta(k));
    s = surf(X, Y, Z, squeeze(PolarTF(:, k, :))', 'EdgeColor', 'none');
    set(s,'facealpha',0.8);
    
    colormap gray;
end

Z = DelayDistance' * ones(1, length(theta));
X = ones(length(DelayDistance), 1) * fr(40) * cos(theta);
Y = ones(length(DelayDistance), 1) * fr(40) * sin(theta);
s = surf(X, Y, Z, squeeze(PolarTF(40, :, :))', 'EdgeColor', 'none');
set(s,'facealpha',0.6);


set(gcf,'Units','centimeter','Position', [1 1 16 20])
view([37.5,10]);

xlim([-max(fr), max(fr)]);
ylim([-max(fr), max(fr)]);
axis off;

%%
N_transducer = length(Wavefront);
transducer_theta = (1:N_transducer)/N_transducer * 2 * pi;

Wavefront = 3e-5 * cos(transducer_theta) -...
            3e-5 * sin(transducer_theta) +...
            1e-4 * cos(2 * transducer_theta);

figure;
hold on;
axis equal;

grid_step   = 0.0001;
text_shift  = 0.00004;
max_line    = (ceil(max(Wavefront)  / grid_step) + 1);
min_line    = (floor(min(Wavefront) / grid_step) - 1);

zero_line   = max(2 * (max_line - min_line), 10);

x = cos(transducer_theta) * zero_line * grid_step;
y = sin(transducer_theta) * zero_line * grid_step;
plot([x, x(1)], [y, y(1)], 'b','LineWidth', 2);
text(0, zero_line * grid_step + text_shift, '0mm');


for line = min_line:-1
    x = cos(transducer_theta) * (zero_line + line) * grid_step;
    y = sin(transducer_theta) * (zero_line + line) * grid_step;
    plot([x, x(1)], [y, y(1)], '--b');
    text(0, (zero_line + line) * grid_step + text_shift, [num2str(line * grid_step * 1e3),'mm']);
end
for line = 1:max_line
    x = cos(transducer_theta) * (zero_line + line) * grid_step;
    y = sin(transducer_theta) * (zero_line + line) * grid_step;
    plot([x, x(1)], [y, y(1)], '--b');
    text(0, (zero_line + line) * grid_step + text_shift, [num2str(line * grid_step * 1e3),'mm']);
end

x = cos(transducer_theta) .* (Wavefront + zero_line * grid_step);
y = sin(transducer_theta) .* (Wavefront + zero_line * grid_step);
plot([x, x(1)], [y, y(1)], 'r', 'LineWidth', 1);
 
axis off;






