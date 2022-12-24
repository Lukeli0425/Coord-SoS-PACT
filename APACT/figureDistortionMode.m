Step               = 0.00004;
DelayDistance_step = 0.00001;
HalfWindow         = 0.0016;
HalfDelay          = 0.0008;

V_sound            = 1520;
FFT_half_N         = 128;

R_ring        = 0.05;
T_sample      = 1/40e6;

ImageX        = - HalfWindow:Step:HalfWindow;
ImageY        = - HalfWindow:Step:HalfWindow;
DelayDistance = - HalfDelay:DelayDistance_step:HalfDelay;

load('EIR_KWAVE.mat');
t0 = 1520;

N_transducer = 512;
delta_angle  = 2*pi/N_transducer;
angle_transducer = delta_angle * (1:N_transducer);

wavefront_function{1} = zeros(1, N_transducer);
                 
wavefront_function{2} = 3e-5 * cos(angle_transducer) -...
                        3e-5 * sin(angle_transducer) +...
                        1e-4 * cos(2 * angle_transducer);
                 
wavefront_function{3} = 3e-5 * cos(angle_transducer) -...
                        3e-5 * sin(angle_transducer);
                 
wavefront_function{4} = 1e-4 * cos(2 * angle_transducer);

                    
for n = 1:4
    TOF = (R_ring - wavefront_function{n}) / V_sound;

    Sinogram = zeros(length(ht), N_transducer);
    for k = 1:N_transducer
        Sinogram(:, k) = circshift(ht, round(TOF(k) / T_sample - t0));
    end

    NewSinogram  = PhaseCorrectPASignal(Sinogram, ht, t0);
    Sinogram     = NewSinogram;

    %%
    PSF = DelayAndSumReconstruction...
        (R_ring, T_sample, V_sound, Sinogram, ImageX, ImageY); 

    I = imread('FIGURE_FEATURE.bmp');
    P0 = double(I(:, :, 1));
    P0 = (255 - P0) / 255;

    Image = conv2(PSF, P0, 'same');

    figure;
    imagesc(ImageX, ImageY, Image);
    axis square;
    colormap gray;
    set(gca,'YDir','normal');
    axis off;

    figure;
    imagesc(ImageX, ImageY, PSF);
    axis square;
    colormap gray;
    set(gca,'YDir','normal');
    axis off;
    
end