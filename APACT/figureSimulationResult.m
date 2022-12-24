
%load('G:\光声成像\局部傅里叶变换自适应重建算法\BigData\20200510.mat')
load('G:\光声成像\局部傅里叶变换自适应重建算法\Result_20201011_SIMULATION_WITHDERIVATIVE.mat')
%%
%1mm
bar_x = [10*Step, 35*Step] + -1.1e-2 + 2e-3;
bar_y = [10*Step, 10*Step] + -1.1e-2 + 2e-3;

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
caxis([min(min(WholeImage)) , max(max(WholeImage))]);
axis square;
axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;
plot(bar_x, bar_y, 'LineWidth', 3, 'Color', [1,1,1]);

%%

for k = 1:length(Look_x)


    WholeImageCorrected   = JointAnImage(Look_y_id(k), Look_x_id(k), WholeImageCorrected, CorrectedImage{k}, ImageX, ImageY);
    
%     figure(2)
%     imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
%     caxis([min(min(WholeImageCorrected)) , max(max(WholeImageCorrected))] / 2);
%     axis square;
%     axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);
%     colormap gray;
%     set(gca,'YDir','normal');
%     axis off;
%     drawnow;
end

figure;
imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
caxis([min(min(WholeImageCorrected)) , max(max(WholeImageCorrected))]);
axis square;
axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;

%%
figure;
colorbar;
colormap gray;
colorbar('Ticks', [0,1], 'TickLabels',{'Min','Max'});
axis off;

%%
load('G:\光声成像\局部傅里叶变换自适应重建算法\KWAVE_P0_2_R1CM1500.mat');

Sinogram0 = -2 * (Sinogram(2:end,:) - Sinogram(1:end-1,:));
Sinogram0  = PhaseCorrectPASignal(Sinogram0, ht, t0);
WholeImage0 = DelayAndSumReconstruction...
    (R_ring, T_sample, 1500, Sinogram0, WholeImageX, WholeImageY); 

%%
figure;
imagesc(WholeImageX, WholeImageY, WholeImage0);
caxis([min(min(WholeImage0)) , max(max(WholeImage0))]);
axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
caxis([min(min(WholeImage))/2 , max(max(WholeImage))/2]);
axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

figure;
imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
caxis([min(min(WholeImageCorrected))/2 , max(max(WholeImageCorrected))/2]);
axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

%%

I1 = WholeImage / sqrt(sum(sum(WholeImage.^2)));
I2 = WholeImageCorrected / sqrt(sum(sum(WholeImageCorrected.^2)));
I0 = WholeImage0 / sqrt(sum(sum(WholeImage0.^2)));

CompareSSIM
%%
I = imread('P0_2.bmp');
P0 = double(I(:, :, 1));
P0 = (255 - P0)/255;

[Ny, Nx] = size(P0);
dy = 1.1 * 2 * R_ring / Ny;
dx = 1.1 * 2 * R_ring / Nx;

X = dx * ((1:Nx) - round(Nx / 2));
Y = dy * ((1:Ny) - round(Ny / 2));

imagesc(X, Y, flipud(rot90(P0)));

axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

%%
[XX, YY] = meshgrid(X, Y);
R = 1e-2;
R1 = 6e-3;
offset = 2e-3;
SOS = ones(Ny, Nx) * 1500;
SOS(XX.^2 + YY.^2 < R^2) = 1600;
SOS(XX.^2 + (YY + offset).^2 < R1^2) = 1650;

figure;
imagesc(X, Y, flipud(rot90(SOS)));

axis([-1.1e-2, 1.1e-2, -1.1e-2, 1.1e-2]);
axis square;
set(gca,'YDir','normal');
axis off;


%%
load('G:\光声成像\局部傅里叶变换自适应重建算法\WAVEFRONT_ANALYTICAL.mat');

c = cos(2 * direction_q);
s = sin(2 * direction_q);

Wa = cell(length(direction_q), 1);

for k = 1:length(Look_x)

    Wa{k}.dc   = mean(wavefront_analytical{k});
    Wa{k}.ccos = c * wavefront_analytical{k}' / (c*c');
    Wa{k}.csin = s * wavefront_analytical{k}' / (s*s');
end

theta = (0:1000) / 1000 * 2 * pi;
wavefront_function   = cell(length(Look_x), 1);
wavefront_function_a = cell(length(Look_x), 1);

%%
figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;

caxis([min(min(WholeImage)) , max(max(WholeImage))]);

dc   = zeros(length(Look_x), 1);
ccos = zeros(length(Look_x), 1);
csin = zeros(length(Look_x), 1);

dc_a   = zeros(length(Look_x), 1);
ccos_a = zeros(length(Look_x), 1);
csin_a = zeros(length(Look_x), 1);

for k = 1:length(Look_x)
    
    dc(k)   = WavefrontTrial{k_opt(k)}.dc;
    ccos(k) = WavefrontTrial{k_opt(k)}.ccos;
    csin(k) = WavefrontTrial{k_opt(k)}.csin;
    
    dc_a(k)   = Wa{k}.dc;
    ccos_a(k) = Wa{k}.ccos;
    csin_a(k) = Wa{k}.csin;
    
    wavefront_function{k} = dc(k) + ...
                            ccos(k) * cos(2 * theta) +...
                            csin(k) * sin(2 * theta);
                        
    wavefront_function_a{k} = dc_a(k) + ...
                              ccos_a(k) * cos(2 * theta) +...
                              csin_a(k) * sin(2 * theta);
                          
    plot(Look_x(k) + (2.5e-4 + wavefront_function_a{k} / 3) .* cos(theta), ...
         Look_y(k) + (2.5e-4 + wavefront_function_a{k} / 3) .* sin(theta), 'g', 'LineWidth', 1.5);
    if(Res(k) < 0.5)
        plot(Look_x(k) + (2.5e-4 + wavefront_function{k} / 3) .* cos(theta), ...
             Look_y(k) + (2.5e-4 + wavefront_function{k} / 3) .* sin(theta), 'r', 'LineWidth', 1.5);
    else
        plot(Look_x(k) + (2.5e-4 + wavefront_function{k} / 3) .* cos(theta), ...
             Look_y(k) + (2.5e-4 + wavefront_function{k} / 3) .* sin(theta), 'm', 'LineWidth', 1.5);
    end
end

plot(- 0.9e-2 + 2.5e-4 * cos(theta), - 0.9e-2 + 2.5e-4 * sin(theta), '.b');

xlim([-1.1e-2, 1.1e-2]);
ylim([-1.1e-2, 1.1e-2]);


%%
V_water = 1500;

[A0, B0] = reconstructSOSmatrix_small(WholeImageX, WholeImageY, Mask, Look_x, Look_y, dc, ccos, csin, ...
                                        V_sound, V_water, R_ring, Step);

%%
TH = 0.5;
                                    
Noise = kron(Res, [1; 1; 1]);
I = Noise < TH;
Noise = Noise(I);

A = A0(I, :);
B = B0(I, :);

% Gaussian priori

w = gausswin(75);
W = w * w';
A2 = zeros(size(A));
for k = 1:size(A, 1)
    k
    
    AA = zeros(size(Mask));
    AA(Mask) = A(k, :);
    AA = conv2(AA, W, 'same');
    A2(k, :) = AA(Mask);
end

sos = A2' * ((A2 * A' + diag(Noise) * 0.00004)\ B);

SOS = ones(size(Mask)) * V_water;

figure;
SOS(Mask) = V_sound./(V_sound / V_water - sos);
imagesc(WholeImageX, WholeImageY, SOS);
axis square;
set(gca,'YDir','normal');
axis off;

xlim([-1.1e-2, 1.1e-2]);
ylim([-1.1e-2, 1.1e-2]);

caxis([1500, 1650]);

drawnow;










