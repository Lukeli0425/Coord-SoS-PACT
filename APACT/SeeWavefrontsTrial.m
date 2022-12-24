

load('G:\光声成像\局部傅里叶变换自适应重建算法\BigData\SIMULATION_RESULT.mat');

C1 = zeros(3, length(Wavefront_a));
for k = 1:length(Wavefront_a)
    [Wavefront_project, Coefficient] = ProjectWavefront(Wavefront_a{k}, Mode);
    C1(:, k) = Coefficient;
end

figure;
scatter3(C1(1, :), C1(2, :), C1(3, :));

figure;
hold on;
plot(C1(1, :));
plot(C1(2, :));
plot(C1(3, :));

%%
Dc  = -16e-5:4e-5:16e-5;
Amp =   0e-5:4e-5:28e-5;
Phi = (1:32) / 32 * pi;

[WavefrontTrial, AngularFrequency] = GenerateWavefrontTrials(Dc, Phi, Amp, Step, FFT_half_N);


C2 = zeros(3, length(WavefrontTrial));
for k = 1:length(WavefrontTrial)
    [Wavefront_project, Coefficient] = ProjectWavefront(WavefrontTrial{k}.W, Mode);
    C2(:, k) = Coefficient;
end

figure;
hold on;
scatter3(C1(1, :), C1(2, :), C1(3, :));
scatter3(C2(1, :), C2(2, :), C2(3, :));

% figure;
% hold on;
% plot(C2(1, :));
% plot(C2(2, :));
% plot(C2(3, :));

