R_ring        = 0.05;
Step          = 0.00004;
V_sound       = 1520;
HalfWindow    = 0.0016;

ImageX        = - HalfWindow:Step:HalfWindow;
ImageY        = - HalfWindow:Step:HalfWindow;

WholeImageX = -0.015: Step: 0.015;
WholeImageY = -0.015: Step: 0.015;

[WholeImageXX, WholeImageYY] = meshgrid(WholeImageX, WholeImageY);
Mask = WholeImageXX.^2 + WholeImageYY.^2 < 1e-2^2;
[Look_x, Look_y, Look_x_id, Look_y_id] =...
    FormPatchCenters(WholeImageX, WholeImageY, Mask, ImageX, ImageY);


R1 = 1e-2;
R2 = 6e-3;
offset = 2e-3;
V_water =  1500;
V_media_1 = 1600;
V_media_2 = 1650;

wavefront_analytical = cell(1, length(Look_x));
W = cell(1, length(Look_x));
direction_end =  cell(1, length(Look_x));
refracted = cell(1, length(Look_x));
%%

for k = 1:length(Look_x)
    k
    [W{k}, direction_end{k}, refracted{k}, wavefront_analytical{k}] = SnellWavefrontAnalytical(Look_x(k), Look_y(k),...
                                    R1, R2, [0, 0], [- offset, 0], R_ring,...
                                    V_water, V_media_1, V_media_2, V_sound);
end

direction_q = ((1:512) / 512) * 2 * pi;

%%
figure;
axis square;
hold on;
for k = 1:length(wavefront_analytical)
    

%     plot(Look_x(k) + (3e-4 + W{k}(refracted{k} > 0) / 10) .* cos(direction_end{k}(refracted{k} > 0)), ...
%          Look_y(k) + (3e-4 + W{k}(refracted{k} > 0) / 10) .* sin(direction_end{k}(refracted{k} > 0)), 'r');
% 
%     plot(Look_x(k) + (3e-4 + W{k}(refracted{k} < 1) / 10) .* cos(direction_end{k}(refracted{k} < 1)), ...
%          Look_y(k) + (3e-4 + W{k}(refracted{k} < 1) / 10) .* sin(direction_end{k}(refracted{k} < 1)), 'b');
     
    plot(Look_x(k) + (2.5e-4 + wavefront_analytical{k} / 3) .* cos(direction_q), ...
         Look_y(k) + (2.5e-4 + wavefront_analytical{k} / 3) .* sin(direction_q), 'g');
end

plot(- 0.9e-2 + 2.5e-4 * cos(direction_q), - 0.9e-2 + 2.5e-4 * sin(direction_q), '.b');

xlim([-1.1e-2, 1.1e-2]);
ylim([-1.1e-2, 1.1e-2]);


%%

load('WAVEFRONT_ANALYTICAL.mat')

%%
figure;
axis square;
hold on;
for k = 1:length(wavefront_analytical)
    

    plot(Look_x(k) + (2.5e-4 + wavefront{k} / 3) .* cos(theta'), ...
         Look_y(k) + (2.5e-4 + wavefront{k} / 3) .* sin(theta'), 'g');
     
    plot(Look_x(k) + (2.5e-4 + wavefront_analytical{k} / 3) .* cos(direction_q), ...
         Look_y(k) + (2.5e-4 + wavefront_analytical{k} / 3) .* sin(direction_q), 'y');
end

plot(- 0.9e-2 + 2.5e-4 * cos(direction_q), - 0.9e-2 + 2.5e-4 * sin(direction_q), '.b');

xlim([-1.1e-2, 1.1e-2]);
ylim([-1.1e-2, 1.1e-2]);


