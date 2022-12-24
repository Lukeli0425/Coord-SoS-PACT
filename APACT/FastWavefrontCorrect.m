function [CorrectedFT, Residual, k_opt] = FastWavefrontCorrect(FTVolume, AngularFrequency)

    %A * X = B ---- AH * A * X = AH * B
    %  X   = AH * B / (AH * A)
    %  Res = BH * B - (AH * B)H * X
    
    B = zeros(size(FTVolume));
    
    for d = 1:size(FTVolume, 3)
        B(:, :, d) = FTVolume(:, :, d) .* AngularFrequency;
    end
    
    BHB = sum(B .* conj(B), 3);
    
    Res = zeros(1930, 1);
    parfor k = 1:1930
        t = load(['Transfer Function2\TF', num2str(k), '.mat']);
        AHB    = sum(t.TF.AH .* B, 3);
        X      = AHB ./ t.TF.AHA;
        Res(k) = sum(sum((BHB - conj(AHB) .* X)));
        k
    end
    
    [Residual, k_opt] = min(Res);
    
    t = load(['Transfer Function2\TF', num2str(k_opt), '.mat']);
    AHB           = sum(t.TF.AH .* B, 3);
    CorrectedFT   = AHB ./ t.TF.AHA;

    Residual = Residual / sum(sum(BHB));
end
