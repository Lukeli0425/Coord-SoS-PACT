load('LIVER_RESULT.mat');

x_left  = -1.1e-2;
x_right = 1.3e-2;
y_down  = -1.2e-2;
y_up = 1.1e-2;

%1mm
bar_x = [10*Step, 35*Step] + x_left + 2e-3;
bar_y = [10*Step, 10*Step] + y_down + 2e-3;

%%

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
caxis([min(min(WholeImage))/4 , max(max(WholeImage))/4]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;
axis([x_left, x_right, y_down, y_up]);
plot(bar_x, bar_y, 'LineWidth', 3, 'Color', [1,1,1]);

[xzoom, yzoom] = ginput(2); 

rectangle('Position',[xzoom(1) - 0.2e-2, yzoom(1) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','r','LineWidth',3 , 'LineStyle', '--');
rectangle('Position',[xzoom(2) - 0.2e-2, yzoom(2) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','b','LineWidth',3 , 'LineStyle', '--');


figure;
imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
caxis([min(min(WholeImageCorrected))/4 , max(max(WholeImageCorrected))/4]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
axis([x_left, x_right, y_down, y_up]);

rectangle('Position',[xzoom(1) - 0.2e-2, yzoom(1) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','r','LineWidth',3 , 'LineStyle', '--');
rectangle('Position',[xzoom(2) - 0.2e-2, yzoom(2) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','b','LineWidth',3 , 'LineStyle', '--');

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
caxis([min(min(WholeImage))/4 , max(max(WholeImage))/4]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;
rectangle('Position',[xzoom(1) - 0.2e-2, yzoom(1) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','r','LineWidth',3 , 'LineStyle', '--');
rectangle('Position',[xzoom(2) - 0.2e-2, yzoom(2) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','b','LineWidth',3 , 'LineStyle', '--');
axis([xzoom(1) - 0.2e-2, xzoom(1) + 0.2e-2, yzoom(1) - 0.2e-2, yzoom(1) + 0.2e-2]);

figure;
imagesc(WholeImageX, WholeImageY, WholeImage);
caxis([min(min(WholeImage))/4 , max(max(WholeImage))/4]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
hold on;
rectangle('Position',[xzoom(1) - 0.2e-2, yzoom(1) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','r','LineWidth',3 , 'LineStyle', '--');
rectangle('Position',[xzoom(2) - 0.2e-2, yzoom(2) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','b','LineWidth',3 , 'LineStyle', '--');
axis([xzoom(2) - 0.2e-2, xzoom(2) + 0.2e-2, yzoom(2) - 0.2e-2, yzoom(2) + 0.2e-2]);

figure;
imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
caxis([min(min(WholeImageCorrected))/4 , max(max(WholeImageCorrected))/4]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
rectangle('Position',[xzoom(1) - 0.2e-2, yzoom(1) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','r','LineWidth',3 , 'LineStyle', '--');
rectangle('Position',[xzoom(2) - 0.2e-2, yzoom(2) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','b','LineWidth',3 , 'LineStyle', '--');
axis([xzoom(1) - 0.2e-2, xzoom(1) + 0.2e-2, yzoom(1) - 0.2e-2, yzoom(1) + 0.2e-2]);

figure;
imagesc(WholeImageX, WholeImageY, WholeImageCorrected);
caxis([min(min(WholeImageCorrected))/4 , max(max(WholeImageCorrected))/4]);
axis square;
colormap gray;
set(gca,'YDir','normal');
axis off;
rectangle('Position',[xzoom(1) - 0.2e-2, yzoom(1) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','r','LineWidth',3 , 'LineStyle', '--');
rectangle('Position',[xzoom(2) - 0.2e-2, yzoom(2) - 0.2e-2, 0.4e-2, 0.4e-2],'EdgeColor','b','LineWidth',3 , 'LineStyle', '--');
axis([xzoom(2) - 0.2e-2, xzoom(2) + 0.2e-2, yzoom(2) - 0.2e-2, yzoom(2) + 0.2e-2]);

%%

figure;
colorbar;
colormap gray;
colorbar('Ticks', [0,1], 'TickLabels',{'Min','Max'});
axis off;


%%
load('RING_ERROR_OLD.mat');

transducer_x = R_ring * cos(transducer_theta);
transducer_y = R_ring * sin(transducer_theta);

TH = 0.7;

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
    
    theta_sample = zeros(length(transducer_x), 1);
    for t = 1:length(transducer_x)
        dx = transducer_x(t) - Look_x(k);
        dy = transducer_y(t) - Look_y(k);
        d  = norm([dx, dy]);
        if(dy < 0)
            theta_sample(t) = acos(dx / d);
        else
            theta_sample(t) = 2 * pi - acos(dx / d);
        end
    end
    
    w_error = - V_sound / V_water * interp1(theta_sample, e, theta, 'linear','extrap');
    
    dc_e   = mean(w_error);
    ccos_e = w_error * c / (c' * c);
    csin_e = w_error * s / (s' * s);
    
    dc(k)   = WavefrontTrial{k_opt(k)}.dc   - dc_e;
    ccos(k) = WavefrontTrial{k_opt(k)}.ccos - ccos_e;
    csin(k) = WavefrontTrial{k_opt(k)}.csin - csin_e;
    
    wavefront_function{k} = dc(k) + ...
                            ccos(k) * cos(2 * theta_plot) +...
                            csin(k) * sin(2 * theta_plot);
                        
                           
    if(Res(k) < TH)
        plot(Look_x(k) + (2.5e-4 + wavefront_function{k} / 3) .* cos(theta_plot), ...
             Look_y(k) + (2.5e-4 + wavefront_function{k} / 3) .* sin(theta_plot), 'r');
    end
   
end

plot(- 0.9e-2 + 2.5e-4 * cos(theta_plot), - 0.9e-2 + 2.5e-4 * sin(theta_plot), 'b', 'LineWidth', 1);



%%
% List the equations

V_water = waterSOSfromTemperature(31);

[A0, B0] = reconstructSOSmatrix_small(WholeImageX, WholeImageY, Mask, Look_x, Look_y, dc, ccos, csin, ...
                                        V_sound, V_water, R_ring, Step);

%%
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

axis([x_left, x_right, y_down, y_up]);
drawnow;








