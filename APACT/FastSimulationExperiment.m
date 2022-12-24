close all;
clear all;

load('KWAVE_P0_2_R1CM1500_1600_1650.mat');

Delay_id = 1;
load('EIR_KWAVE.mat');
t0 = 1520;

%Use differential term
Sinogram = -2 * (Sinogram(2:end,:) - Sinogram(1:end-1,:));
ht       = -2 * (ht(2:end) - ht(1:end-1));

R_ring        = 0.05;
T_sample      = 1/40e6;
Sinogram      = Sinogram(Delay_id : end, :);

Step               = 0.00004;
DelayDistance_step = 0.00001;
HalfWindow         = 0.0016;
HalfDelay          = 0.0008;

V_sound            = 1500;%= 1520;

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
%caxis([min(min(WholeImage))/2, max(max(WholeImage))/2]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;

[WholeImageXX, WholeImageYY] = meshgrid(WholeImageX, WholeImageY);
Mask = WholeImageXX.^2 + WholeImageYY.^2 < 1e-2^2;
[Look_x, Look_y, Look_x_id, Look_y_id] =...
    FormPatchCenters(WholeImageX, WholeImageY, Mask, ImageX, ImageY);

% hold on;
% scatter(Look_x, Look_y);

%%

% Dc  = -16e-5:4e-5:16e-5;
% Amp =   0e-5:4e-5:28e-5;
% Phi = (1:32) / 32 * pi;
%[WavefrontTrial, AngularFrequency] = GenerateWavefrontTrials(Dc, Phi, Amp, Step, FFT_half_N);
[WavefrontTrial, AngularFrequency] = GenerateWavefrontTrials_lattice([-2e-4 1.6e-4], 3.2e-4, 4e-5, Step, FFT_half_N);

% tic
% for k = 1:length(WavefrontTrial)
%     k / length(WavefrontTrial)
%     TF = PrepareFastWavefrontCorrect(WavefrontTrial{k}.W, AngularFrequency, FFT_half_N, DelayDistance);
%     save(['C:\Users\Administrator\Desktop\cuimanxiu\自适应重建\Transfer Function2\TF', num2str(k),'.mat'], 'TF');
% end
% toc

%%
parpool(32);

tic

Res = zeros(length(Look_x), 1);
k_opt = zeros(length(Look_x), 1);
Wavefront = cell(length(Look_x),  1);
CorrectedImage = cell(length(Look_x),  1);

parfor k = 1:length(Look_x)%410:412%388:390%
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

    WholeImageCorrected   = JointAnImage(Look_y_id(k), Look_x_id(k), WholeImageCorrected, CorrectedImage{k} , ImageX, ImageY);
   
    figure(3)
    imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
    axis square;
    colormap gray;
    set(gca,'YDir','normal');
    caxis([min(min(WholeImageCorrected))/2, max(max(WholeImageCorrected))/2]);
    axis off;
    drawnow;
    
%     figure;
%     imagesc(ImageX, ImageY, CorrectedImage{k});
%     axis square;
%     colormap gray;
%     set(gca,'YDir','normal');
%     %caxis([min(min(WholeImageCorrected))/2, max(max(WholeImageCorrected))/2]);
%     axis off;
%     drawnow;

end

%%
figure;
hold on;
plot(Res);
