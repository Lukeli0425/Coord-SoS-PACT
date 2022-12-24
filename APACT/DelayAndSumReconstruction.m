function Image = DelayAndSumReconstruction...
    (R_ring, T_sample, V_sound, Sinogram, ImageX, ImageY)
% DESCRIPTION:
%     Generate a 2D Delay And Sum reconstructed PACT image of ring
%     transducer array
%
% INPUTS:
%     R_ring          - The radius [m] of the ring transducer array.
%     T_sample        - Sample time interval of the signals [s].
%     V_sound         - The sound speed [m/s] used for Delay And Sum 
%                     reconstruction.
%     ImageX          - The vector [m] defining the x coordinates of the
%                     grid points on which the reconstruction is done. The
%                     values in the vector should be unifromly-spaced in
%                     ascending order. The origin of the cartesian
%                     coordinate system is the center of the ring array.
%     ImageY          - The vector [m] defining the y coordinates of the
%                     grid points on which the reconstruction is done. The
%                     values in the vector should be unifromly-spaced in
%                     ascending order.
%     Sinogram        - A 2D matrix and each column of it is the signal
%                     recievde by one transducer. The nummber of
%                     transducers should be the number of columns. The
%                     transducers should be evenly distributed on a circle
%                     in counterclockwise arrangement and the first column
%                     correspond to the transducer in the dirrection 1 *
%                     2pi/N in the first quartile. The first sample should
%                     be at time 0 when the photoacoustic effect happens.
%
% OUTPUTS:
%     Image           - A matrix whose size is length(ImageY) times
%                     length(ImageX). Image(t, s) is the reconstructed
%                     photoacoustic amplitude at the grid point (ImageX(s),
%                     ImageY(t)).
% ABOUT:
%     author          - Manxiu Cui
%     last update     - 25th September 2019
%
% This function is part of the Local Fourier Domain Photoacoustic
% Tomography Wavefront Correction Project 
%
% Copyright (C) 2019 Caltech Visiting Undergraduate Research Program 
% Manxiu Cui, Tsinghua University, Department of EE

    N_transducer = size(Sinogram, 2);
    
    Image = zeros(length(ImageX), length(ImageY));
    delta_angle = 2*pi/N_transducer;
    angle_transducer = delta_angle * (1:N_transducer);
    
    x_transducer = R_ring * cos(angle_transducer);
    y_transducer = R_ring * sin(angle_transducer);
    
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