function NewSinogram = PhaseCorrectPASignal(Sinogram, EIR, t0)

    Delta = zeros(size(EIR));
    Delta(t0) = 1;
    
    Delta_ft    = fft(Delta);
    EIR_ft      = fft(EIR);
    Sinogram_ft = fft(Sinogram);
    
    Sinogram_ft = Sinogram_ft.* ( ...
                    exp(1i * (angle(Delta_ft) - angle(EIR_ft))) * ...
                    ones(1, size(Sinogram_ft, 2))...
                                );
    
    NewSinogram = real(ifft(Sinogram_ft));
end