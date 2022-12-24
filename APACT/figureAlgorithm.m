clear all;

%load('G:\光声成像\局部傅里叶变换自适应重建算法\BigData\20200510.mat');
load('G:\光声成像\局部傅里叶变换自适应重建算法\Result_20201011_SIMULATION_WITHDERIVATIVE.mat');
load('G:\光声成像\局部傅里叶变换自适应重建算法\WAVEFRONT_ANALYTICAL.mat')


%% The figure 2 (a)
k0 = 149;

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;

caxis([min(min(WholeImage)), max(max(WholeImage))]);

axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);

north = max(ImageY) + Look_y(k0);
east  = max(ImageX) + Look_x(k0);
west  = min(ImageX) + Look_x(k0);
south = min(ImageY) + Look_y(k0);

plot([west, east], [north, north], 'r--', 'LineWidth', 2);
plot([west, east], [south, south], 'r--', 'LineWidth', 2);
plot([west, west], [south, north], 'r--', 'LineWidth', 2);
plot([east, east], [south, north], 'r--', 'LineWidth', 2);

north = max(ImageY) + Look_y(k0 + 1);
east  = max(ImageX) + Look_x(k0 + 1);
west  = min(ImageX) + Look_x(k0 + 1);
south = min(ImageY) + Look_y(k0 + 1);

plot([west, east], [north, north], 'b--', 'LineWidth', 2);
plot([west, east], [south, south], 'b--', 'LineWidth', 2);
plot([west, west], [south, north], 'b--', 'LineWidth', 2);
plot([east, east], [south, north], 'b--', 'LineWidth', 2);


%% The figure 2 (a) for corrected image
figure;
imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;

caxis([min(min(WholeImageCorrected))/2, max(max(WholeImageCorrected))/2]);

axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);

north = max(ImageY) + Look_y(k0);
east  = max(ImageX) + Look_x(k0);
west  = min(ImageX) + Look_x(k0);
south = min(ImageY) + Look_y(k0);

plot([west, east], [north, north], 'r--', 'LineWidth', 2);
plot([west, east], [south, south], 'r--', 'LineWidth', 2);
plot([west, west], [south, north], 'r--', 'LineWidth', 2);
plot([east, east], [south, north], 'r--', 'LineWidth', 2);

north = max(ImageY) + Look_y(k0 + 1);
east  = max(ImageX) + Look_x(k0 + 1);
west  = min(ImageX) + Look_x(k0 + 1);
south = min(ImageY) + Look_y(k0 + 1);

plot([west, east], [north, north], 'b--', 'LineWidth', 2);
plot([west, east], [south, south], 'b--', 'LineWidth', 2);
plot([west, west], [south, north], 'b--', 'LineWidth', 2);
plot([east, east], [south, north], 'b--', 'LineWidth', 2);


%% figure 2(c) the stack of delayed images

[ImageVolume, FTVolume] = FastGenerate3DVolume...
    (Sinogram, R_ring, T_sample, V_sound, ImageX + Look_x(k0), ImageY + Look_y(k0), DelayDistance, FFT_half_N);


figure;
hold on;
s = slice(ImageX, ImageY, DelayDistance, ...
          ImageVolume, [], [], DelayDistance(1:20:end));
colormap gray;
set(s,'EdgeColor','none');

zlabel('d');  

set(gcf,'Units','centimeter','Position',[1 1 16 20])
view([37.5,10]);

axis off;


north = max(ImageY);
east  = max(ImageX);
west  = min(ImageX);
south = min(ImageY);

plot3([west, east], [north, north], [min(DelayDistance), min(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([west, east], [south, south], [min(DelayDistance), min(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([west, west], [south, north], [min(DelayDistance), min(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([east, east], [south, north], [min(DelayDistance), min(DelayDistance)], 'r--', 'LineWidth', 2);

plot3([west, east], [north, north], [max(DelayDistance), max(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([west, east], [south, south], [max(DelayDistance), max(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([west, west], [south, north], [max(DelayDistance), max(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([east, east], [south, north], [max(DelayDistance), max(DelayDistance)], 'r--', 'LineWidth', 2);

plot3([west, west], [south, south], [min(DelayDistance), max(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([east, east], [south, south], [min(DelayDistance), max(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([west, west], [north, north], [min(DelayDistance), max(DelayDistance)], 'r--', 'LineWidth', 2);
plot3([east, east], [north, north], [min(DelayDistance), max(DelayDistance)], 'r--', 'LineWidth', 2);


%% figure 2(b) PSF formation: prepare

N_transducer = 512;
delta_angle  = 2*pi/N_transducer;
angle_transducer = delta_angle * (1:N_transducer);

wavefront_function = interp1(direction_q, wavefront_analytical{k0}, angle_transducer, 'linear','extrap');

TOF = (R_ring - wavefront_function) / V_sound;

Sinogram0 = zeros(length(ht), N_transducer);
for k = 1:N_transducer
    Sinogram0(:, k) = circshift(ht, round(TOF(k) / T_sample - t0));
end

NewSinogram  = PhaseCorrectPASignal(Sinogram0, ht, t0);
Sinogram0     = NewSinogram;

[PSFVolume, ~] = FastGenerate3DVolume...
    (Sinogram0, R_ring, T_sample, V_sound, ImageX, ImageY, DelayDistance, FFT_half_N);


%% figure 2(b) PSF formation: draw left

close all;

I = zeros(length(ImageY), length(ImageX), 3);

m = min(min(min(ImageVolume)));

for k = 1:40:length(DelayDistance)
    figure;
    
    P = zeros(size(PSFVolume(:, :, k)) + 100);
    P(1:size(PSFVolume(:, :, k), 1), 1:size(PSFVolume(:, :, k), 2)) = PSFVolume(:, :, k);
    P = circshift(circshift(P, -27, 1), -1, 2);
    P = P(1:size(PSFVolume(:, :, k), 1), 1:size(PSFVolume(:, :, k), 2));
    
    I(:, :, 1) = flipud(ImageVolume(:, :, k) - m) * 130;
    I(:, :, 2) = flipud(ImageVolume(:, :, k) - m) * 130 + flipud(P) * 200;
    I(:, :, 3) = flipud(ImageVolume(:, :, k) - m) * 130;
    imshow(I);
    drawnow;
end


%% figure 2(b) PSF formation: draw right

R_exa = 0.005;

angle = 0:0.001:2*pi;
Cos = cos(angle);
Sin = sin(angle);

x_transducer = R_exa * cos(angle_transducer);
y_transducer = R_exa * sin(angle_transducer);

c = cos(direction_q);
s = sin(direction_q);

c1 = c * wavefront_analytical{k0}' / (c*c');
s1 = s * wavefront_analytical{k0}' / (s*s');


for p = 1:40:length(DelayDistance) 
    
    figure;
    
    for k = 32:32:length(angle_transducer)
       %compensate for 1st
       draw_x = x_transducer(k) + (R_exa - wavefront_function(k) + cos(angle_transducer(k)) * c1 + sin(angle_transducer(k)) * s1 + DelayDistance(p)) * Cos;
       draw_y = y_transducer(k) + (R_exa - wavefront_function(k) + cos(angle_transducer(k)) * c1 + sin(angle_transducer(k)) * s1 + DelayDistance(p)) * Sin;
       
       if(k == 256)
           plot(draw_x, draw_y, 'g', 'LineWidth', 8);
       else
           plot(draw_x, draw_y, 'g', 'LineWidth', 4);
       end
       axis([-0.001, 0.001, -0.001, 0.001]);
       hold on;
       axis equal;
    end
    axis off;
    hold off;
    drawnow;
    
end


%% figure (d)(e)(f)(g)(h) prepare: Fourier domain

f = ((1:FFT_half_N * 2) - FFT_half_N - 1)/(2 * FFT_half_N) * 1/Step * 2 * pi;
[AngularFrequency_x, AngularFrequency_y] = meshgrid(f);

AngularFrequency = sqrt(AngularFrequency_x.^2 + AngularFrequency_y.^2);

Theta = acos(AngularFrequency_x ./ AngularFrequency);
Theta(AngularFrequency_y < 0) = - Theta(AngularFrequency_y < 0);

Wavefront = WavefrontTrial{k_opt(k0)}.dc + ...
            WavefrontTrial{k_opt(k0)}.ccos * cos(2 * Theta) +...
            WavefrontTrial{k_opt(k0)}.csin * sin(2 * Theta);


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

fr = (0:60) /(2 * FFT_half_N) * 1/Step * 2 * pi;

Fxq = fr' * cos(angle_transducer);
Fyq = fr' * sin(angle_transducer);

PolarFT = zeros(length(fr), length(angle_transducer), length(DelayDistance));
PolarTF = zeros(length(fr), length(angle_transducer), length(DelayDistance));

for r = 1:length(DelayDistance)
    FTVolume(:, :, r) = FTVolume(:, :, r).* AngularFrequency;
end

for d = 1:length(DelayDistance)
    PolarFT(:, :, d) = interp2(f, f, abs(FTVolume(:, :, d)), Fxq, Fyq);
    PolarTF(:, :, d) = interp2(f, f, abs(T(:, :, d)), Fxq, Fyq);
end

%% figure 2(d)
figure;
hold on;
s = slice(f, f, DelayDistance, ...
          abs(FTVolume), [], [], DelayDistance(1:20:end));
colormap gray;
set(s,'EdgeColor','none');

set(s,'facealpha',0.3);

Z = DelayDistance' * ones(1, length(fr));
for k = 1:32:384
    X = ones(length(DelayDistance), 1) * fr * cos(angle_transducer(k));
    Y = ones(length(DelayDistance), 1) * fr * sin(angle_transducer(k));
    s = surf(X, Y, Z, squeeze(PolarFT(:, k, :))', 'EdgeColor', 'none');
    set(s,'facealpha',1);
    
    colormap gray;
end


set(gcf,'Units','centimeter','Position', [1 1 16 20])
view([37.5,10]);

xlim([-max(fr), max(fr)]);
ylim([-max(fr), max(fr)]);
axis off;

%% figure 2(f)
figure;
hold on;

Z = DelayDistance' * ones(1, length(fr));
for k = 1:32:384
    X = ones(length(DelayDistance), 1) * fr * cos(angle_transducer(k));
    Y = ones(length(DelayDistance), 1) * fr * sin(angle_transducer(k));
    s = surf(X, Y, Z, squeeze(PolarTF(:, k, :))', 'EdgeColor', 'none');
    set(s,'facealpha',0.8);
    
    colormap gray;
end

Z = DelayDistance' * ones(1, length(angle_transducer));
X = ones(length(DelayDistance), 1) * fr(40) * cos(angle_transducer);
Y = ones(length(DelayDistance), 1) * fr(40) * sin(angle_transducer);
s = surf(X, Y, Z, squeeze(PolarTF(40, :, :))', 'EdgeColor', 'none');
set(s,'facealpha',0.6);


set(gcf,'Units','centimeter','Position', [1 1 16 20])
view([37.5,10]);

xlim([-max(fr), max(fr)]);
ylim([-max(fr), max(fr)]);
axis off;

%% figure 2(e)(g)

figure;
imagesc(ImageX, ImageY, CorrectedImage{k0});
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;

figure;
imagesc(f, f, abs(Myfft(CorrectedImage{k0}, FFT_half_N)).* AngularFrequency);
axis square;
colormap gray;
set(gca,'YDir','normal');

axis off;
hold on;

xlim([-max(fr), max(fr)]);
ylim([-max(fr), max(fr)]);

view([37.5,10]);

axis off;



%% figure 2(h)

wavefront_estimate = WavefrontTrial{k_opt(k0)}.dc + ...
                     WavefrontTrial{k_opt(k0)}.ccos * cos(2 * angle_transducer) +...
                     WavefrontTrial{k_opt(k0)}.csin * sin(2 * angle_transducer);

figure;
hold on;
axis equal;

fontSize = 15;

grid_step   = 0.0002;
text_shift  = 0.00005;
max_line    = (ceil(max(wavefront_function)  / grid_step));
min_line    = (floor(min(wavefront_function) / grid_step));

zero_line   = 5;%max(2 * (max_line - min_line), 10);

x = cos(angle_transducer) * zero_line * grid_step;
y = sin(angle_transducer) * zero_line * grid_step;
plot([x, x(1)], [y, y(1)], 'b','LineWidth', 2);
text(text_shift, zero_line * grid_step + text_shift, '0mm', 'FontSize',fontSize);


for line = min_line+1:-1
    x = cos(angle_transducer) * (zero_line + line) * grid_step;
    y = sin(angle_transducer) * (zero_line + line) * grid_step;
    plot([x, x(1)], [y, y(1)], '--b', 'LineWidth', 1);
    text(text_shift, (zero_line + line) * grid_step + text_shift, [num2str(line * grid_step * 1e3),'mm'], 'FontSize',fontSize);
end
for line = 1:max_line-1
    x = cos(angle_transducer) * (zero_line + line) * grid_step;
    y = sin(angle_transducer) * (zero_line + line) * grid_step;
    plot([x, x(1)], [y, y(1)], '--b', 'LineWidth', 1);
    text(text_shift, (zero_line + line) * grid_step + text_shift, [num2str(line * grid_step * 1e3),'mm'], 'FontSize',fontSize);
end

% x = cos(angle_transducer) .* (wavefront_function + zero_line * grid_step);
% y = sin(angle_transducer) .* (wavefront_function + zero_line * grid_step);
% plot([x, x(1)], [y, y(1)], 'g', 'LineWidth', 2);


x = cos(angle_transducer) .* (wavefront_estimate + zero_line * grid_step);
y = sin(angle_transducer) .* (wavefront_estimate + zero_line * grid_step);
plot([x, x(1)], [y, y(1)], 'r', 'LineWidth', 2);
 


axis off;

