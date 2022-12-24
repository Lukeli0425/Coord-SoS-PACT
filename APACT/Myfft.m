function FT = Myfft(I, half_N)

    I_zeropad = zeros(2 * half_N, 2 * half_N);

    [h, w] = size(I);
    idxc = round(w / 2);
    idyc = round(h / 2);

    I_zeropad(1:h, 1:w) = I;
    
    I_zeropad = circshift(I_zeropad, -idyc, 1);
    I_zeropad = circshift(I_zeropad, -idxc, 2);

    FT = fftshift(fft2(I_zeropad));
    
    FT(half_N + 1, half_N + 1) = 0;
end