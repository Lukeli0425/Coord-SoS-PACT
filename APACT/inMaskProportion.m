function P = inMaskProportion(Step, Mask, x_in1, y_in1, x_in2, y_in2)

    N = 100;

    x = x_in1 + (x_in2 - x_in1) * (0:N) / N;
    y = y_in1 + (y_in2 - y_in1) * (0:N) / N;

    I = 0;
    for k = 1:length(x)
          I = I + Mask(round((y(k) + 0.015) / Step) + 1, ...
                       round((x(k) + 0.015) / Step) + 1);
    end

    if(I == 0)
        P = 0;
    elseif(I == N + 1)
        P = 1;
    else
        P = (I - 1/2) / (N - 1);
    end
    
end