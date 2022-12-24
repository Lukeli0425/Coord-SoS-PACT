function TF = PrepareFastWavefrontCorrect(Wavefront, AngularFrequency, FFT_half_N, DelayDistance)

    Wavefront_backward = rot90(Wavefront, 2);
    Wavefront_backward = circshift(Wavefront_backward, 1, 1);
    Wavefront_backward = circshift(Wavefront_backward, 1, 2);

    T   = zeros([size(Wavefront), length(DelayDistance)]);

    for t = 1:length(DelayDistance)
        A_forward  = exp(-1i * AngularFrequency.* (- Wavefront + DelayDistance(t)));
        A_backward = exp( 1i * AngularFrequency.* (- Wavefront_backward + DelayDistance(t)));
        T(:, :, t) = (A_forward + A_backward) .* AngularFrequency / 2;
        T(FFT_half_N + 1, FFT_half_N + 1, t) = 1;
    end

    TF.AH   = conj(T);
    TF.AHA  = sum(T .* TF.AH, 3);
    
end