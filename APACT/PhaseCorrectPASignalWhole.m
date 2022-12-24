function NewSinogram = PhaseCorrectPASignalWhole(Sinogram, Sinogram_ps, x_ps, y_ps, R_ring, V_sound, T_sample)

    N_transducer = size(Sinogram, 2);
    
    delta_angle = 2*pi/N_transducer;
    angle_transducer = delta_angle * (1:N_transducer);
    
    x_transducer = R_ring * cos(angle_transducer);
    y_transducer = R_ring * sin(angle_transducer);

    TOF_id = round(sqrt((x_transducer - x_ps).^2 + (y_transducer - y_ps).^2) / V_sound / T_sample);
    
    Sinogram_delta = zeros(size(Sinogram));
    for n = 1:N_transducer   
        Sinogram_delta(TOF_id(n), n) = 1;
    end
    
    Sinogram_ft       = fft(Sinogram);
    Sinogram_ps_ft    = fft(Sinogram_ps);
    Sinogram_delta_ft = fft(Sinogram_delta);
    
    Sinogram_ft_crct = Sinogram_ft .* exp(1i * (angle(Sinogram_delta_ft) - angle(Sinogram_ps_ft)));
    
    NewSinogram = real(ifft(Sinogram_ft_crct));

    
%     figure;
%     hold on;
%     imagesc(NewSinogram);
%     colormap gray;
%     plot(TOF_id);
%     set(gca,'YDir','reverse');
%     axis off;
end