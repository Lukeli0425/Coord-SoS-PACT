function [x1, y1, hit, l, angle_normal] = hitCircle(x, y, theta, xc, yc, R, in)

    to_middle   = (xc - x) * cos(theta) + (yc - y) * sin(theta);
    d2 = (xc - x).^2 + (yc - y).^2 - to_middle^2;
    
    if R^2 - d2 < 0
       x1 = x;
       y1 = y;
       hit = 0;
       l = 0;
       angle_normal = 0;
       return;
    else
        half_string = sqrt(abs(R^2 - d2));

        if(in)
            l = to_middle + half_string;
            x1 = x + l * cos(theta);
            y1 = y + l * sin(theta);
            hit = 1;
            angle_normal = angle(x1 - xc + (y1 - yc) * 1i);
        else
            if to_middle < 0
                x1 = x;
                y1 = y;
                hit = 0;
                l = 0;
                angle_normal = 0;
            else
                l = to_middle - half_string;
                x1 = x + l * cos(theta);
                y1 = y + l * sin(theta);
                hit = 1;
                angle_normal = angle(xc - x1 + (yc - y1) * 1i);
            end
        end
    end
    
%     if(abs(imag(l)) > 0)
%         R^2 - d2
%     end
end