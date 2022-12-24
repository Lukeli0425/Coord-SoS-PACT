function Image = arbiarrayDelayAndSumReconstruction...
    (x_transducer, y_transducer, T_sample, V_sound, Sinogram, ImageX, ImageY)


    N_transducer = size(Sinogram, 2);
    
    Image = zeros(length(ImageY), length(ImageX));
    
    related_data = zeros(1, N_transducer);
    
    for s = 1:length(ImageX)
        for t = 1:length(ImageY)
            distance_to_transducer = sqrt((x_transducer - ImageX(s)).^2 + (y_transducer - ImageY(t)).^2);
            for k = 1:N_transducer
                id = round(distance_to_transducer(k)/(V_sound * T_sample));
                if(id > size(Sinogram, 1) || id < 1)
                    related_data(k) = 0;
                else
                    related_data(k) = Sinogram(id, k);
                end
            end
            Image(t, s) = mean(related_data);
        end
    end
end
% 
% load('')
% Delay_id = 64;
% 
% R_ring        = 0.05;
% T_sample      = 1/40e6;
% Sinogram      = Sinogram(Delay_id : end, :);
% 
% Step          = 0.00004;
% 
% ImageX = -0.015: Step: 0.015;
% ImageY = -0.015: Step: 0.015;
% 
% N_transducer = size(Sinogram, 2);
% delta_angle = 2*pi/N_transducer;
% angle_transducer = delta_angle * (1:N_transducer);
% 
% R = R_ring -40e-6 - 120e-6 * cos(2 * angle_transducer);
% 
% x_transducer = R .* cos(angle_transducer);
% y_transducer = R .* sin(angle_transducer);
% 
% Image = arbiarrayDelayAndSumReconstruction...
%     (x_transducer, y_transducer, T_sample, V_sound, Sinogram, ImageX, ImageY);
%
% figure;
% imagesc(ImageX, ImageY, Image);
% caxis([min(min(Image)), max(max(Image))]);
% axis square;
% colormap gray;
% set(gca,'YDir','normal');
% axis off;

