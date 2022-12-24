function Sinogram = kWaveExperiment(P0, SOS, dx, dy, Nx, Ny, sos_background, R_ring, N_transducer, T_sample, N_time)

%P0 and SOS should be the

% A = SOS;
% SOS =  1500 * ones([size(SOS), 30]);
% SOS(:, :, 15) = A;
% 
% A = P0;
% P0 =  zeros([size(P0), 30]);
% for z= 1:30
%     P0(:, :, z) = A;
% end

kgrid = makeGrid(Ny,dy,Nx,dx);
kgrid.t_array = T_sample/2 *(0: 2*N_time - 1);

% P0 = smooth(kgrid, P0);

delta_angle = 2*pi/N_transducer;
angle_transducer = delta_angle * (1:N_transducer);

x_transducer = R_ring * cos(angle_transducer);
y_transducer = R_ring * sin(angle_transducer);

sensor.mask = [x_transducer; y_transducer];

source.p0 = P0;
medium.sound_speed = SOS;
rou = 1000; % density
medium.density = rou * ones(Ny, Nx);
medium.sound_speed_ref = sos_background;

% dt_stability_limit = checkStability(kgrid, medium)

sensor_data = kspaceFirstOrder2D(kgrid,medium,source,sensor);

Sinogram = sensor_data(:, 1:2:end)';

end

