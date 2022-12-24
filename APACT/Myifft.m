function I = Myifft(FT, w, h)
    
    idxc = round(w / 2);
    idyc = round(h / 2);
    
    I_zeropad = real(ifft2(ifftshift(FT)));
    I_zeropad = circshift(I_zeropad, idyc, 1);
    I_zeropad = circshift(I_zeropad, idxc, 2);
    
    I = I_zeropad(1:h, 1:w);
end