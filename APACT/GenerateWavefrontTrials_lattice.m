function [WavefrontTrial, AngularFrequency] = GenerateWavefrontTrials_lattice(dc, amp, step, Step, FFT_half_N)

    f = ((1:FFT_half_N * 2) - FFT_half_N - 1)/(2 * FFT_half_N) * 1/Step * 2 * pi;
    [AngularFrequency_x, AngularFrequency_y] = meshgrid(f);
    
    AngularFrequency = sqrt(AngularFrequency_x.^2 + AngularFrequency_y.^2);
    
    Theta = acos(AngularFrequency_x ./ AngularFrequency);
    Theta(AngularFrequency_y < 0) = - Theta(AngularFrequency_y < 0);
    
    k = 0;
    
    for z = dc(1):step:dc(2)
        for x = - amp:step:amp
            for y = - amp:step:amp
                if(x^2 + y^2 < amp^2)
                    k = k + 1;
                    WavefrontTrial{k}.W = z + x * cos(2 * Theta) + y * sin(2 * Theta);
                    WavefrontTrial{k}.W(FFT_half_N + 1, FFT_half_N + 1) = 0;               

                    WavefrontTrial{k}.dc    = z;
                    WavefrontTrial{k}.ccos  = x;
                    WavefrontTrial{k}.csin  = y;
                end
            end
        end
    end
    
end

    