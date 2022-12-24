close all;
clear all;

%load('ANIMAL_LAYER.mat');
%Delay_id = 44;

% load('G:\光声成像\局部傅里叶变换自适应重建算法\CMX_invivoDATA\kidney\layer40.mat')
% Delay_id = 57;

% load('G:\光声成像\局部傅里叶变换自适应重建算法\CMX_invivoDATA\hepg2cmx\layer5.mat')
% Delay_id = 64;

load('G:\光声成像\局部傅里叶变换自适应重建算法\kidney_selected_20200530\kidney_strong&good.mat')
Delay_id = 64;

load('EIR_TUNGSTEN.mat');
ht = EIR_AVE_20180516_LEFT(Delay_id : end); 
t0 = 1120 - Delay_id + 1; 

R_ring        = 0.05;
T_sample      = 1/40e6;
Sinogram      = Sinogram(Delay_id : end, :);

Step               = 0.00004;
DelayDistance_step = 0.00001;
HalfWindow         = 0.0016;
HalfDelay          = 0.0008;

%liver
%V_sound            = 1515;

%kidney
%V_sound            = 1530;

%hepg2
%V_sound            = 1525;

%kidney strong
%V_sound            = 1525;

%phantom
%V_sound            = 1505;


FFT_half_N         = 128;

ImageX        = - HalfWindow:Step:HalfWindow;
ImageY        = - HalfWindow:Step:HalfWindow;
DelayDistance = - HalfDelay:DelayDistance_step:HalfDelay;

N_transducer = size(Sinogram, 2);
N_time       = size(Sinogram, 1);

NewSinogram  = PhaseCorrectPASignal(Sinogram, ht, t0);
Sinogram     = NewSinogram;

WholeImageX = -0.015: Step: 0.015;
WholeImageY = -0.015: Step: 0.015;

WholeImage = DelayAndSumReconstruction...
    (R_ring, T_sample, V_sound, Sinogram, WholeImageX, WholeImageY); 

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
caxis([min(min(WholeImage))/2 , max(max(WholeImage))/2]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

%%

% [WholeImageXX, WholeImageYY] = meshgrid(WholeImageX, WholeImageY);
% Mask = WholeImageXX.^2 + WholeImageYY.^2 < R^2;
% Mask = roipoly;
% [Look_x, Look_y, Look_x_id, Look_y_id] =...
%     FormPatchCenters(WholeImageX, WholeImageY, Mask, ImageX, ImageY);

load('IN_VIVO_XY.mat');
hold on;
scatter(Look_x, Look_y, '+');

%%
% Dc  = -16e-5:4e-5:16e-5;
% Amp =   0e-5:4e-5:28e-5;
% Phi = (1:32) / 32 * pi;
% 
% [WavefrontTrial, AngularFrequency] = GenerateWavefrontTrials(Dc, Phi, Amp, Step, FFT_half_N);

[WavefrontTrial, AngularFrequency] = GenerateWavefrontTrials_lattice([-2e-4 1.6e-4], 3.2e-4, 4e-5, Step, FFT_half_N);

% tic
% for k = 1:length(WavefrontTrial)
%     k / length(WavefrontTrial)
%     TF = PrepareFastWavefrontCorrect(WavefrontTrial{k}.W, AngularFrequency, FFT_half_N, DelayDistance);
%     save(['C:\Users\Administrator\Desktop\cuimanxiu\自适应重建\Transfer Function\TF', num2str(k),'.mat'], 'TF');
% end
% toc

%%
parpool(32);

tic

Res = zeros(length(Look_x), 1);
k_opt = zeros(length(Look_x), 1);
Wavefront = cell(length(Look_x),  1);
CorrectedImage = cell(length(Look_x),  1);

parfor k = 1:length(Look_x)
    k
    [ImageVolume, FTVolume] = FastGenerate3DVolume...
    (Sinogram, R_ring, T_sample, V_sound, ImageX + Look_x(k), ImageY + Look_y(k), DelayDistance, FFT_half_N);
    
    [CorrectedFT, Res(k), k_opt(k)] = FastWavefrontCorrect(FTVolume, AngularFrequency);
    
    CorrectedImage{k} = Myifft(CorrectedFT, length(ImageX), length(ImageY));
end

toc
%%
WholeImageCorrected = zeros(size(WholeImage));

for k = 1:length(Look_x)

%     figure(2)
%     hold on;
%     imagesc(Look_x(k) + fs, Look_y(k) + fs, Wavefront{k});
%     
%     xlim([-0.015, 0.015]);
%     ylim([-0.015, 0.015]);
%     drawnow;

    WholeImageCorrected   = JointAnImage(Look_y_id(k), Look_x_id(k), WholeImageCorrected, CorrectedImage{k}, ImageX, ImageY);
   
    figure(3)
    imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
    caxis([min(min(WholeImageCorrected))/2 , max(max(WholeImageCorrected))/2]);
    axis square;
    colormap gray;
    set(gca,'YDir','normal');
    axis off;
    drawnow;
end

figure;
hold on;
plot(Res);



%%


theta_plot = (0:1000) / 1000 * 2 * pi;

theta = (1:1000) / 1000 * 2 * pi;

wavefront_function   = cell(length(Look_x), 1);

figure;
axis equal;
set(gca,'YDir','normal');
axis off;
hold on;

dc   = zeros(length(Look_x), 1);
ccos = zeros(length(Look_x), 1);
csin = zeros(length(Look_x), 1);

c = cos(2 * theta');
s = sin(2 * theta');

for k = 1:length(Look_x)
    

    
    dc(k)   = WavefrontTrial{k_opt(k)}.dc;
    ccos(k) = WavefrontTrial{k_opt(k)}.ccos;
    csin(k) = WavefrontTrial{k_opt(k)}.csin;
    
    wavefront_function{k} = dc(k) + ...
                            ccos(k) * cos(2 * theta_plot) +...
                            csin(k) * sin(2 * theta_plot);
                        
                           

    plot(Look_x(k) + (2.5e-4 + wavefront_function{k} / 3) .* cos(theta_plot), ...
         Look_y(k) + (2.5e-4 + wavefront_function{k} / 3) .* sin(theta_plot), 'r');

end


