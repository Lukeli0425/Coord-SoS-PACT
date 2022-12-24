function [W, direction_end, refracted, wavefront_analytical] = SnellWavefrontAnalytical(point_x, point_y,...
                                    R1, R2, Center_1, Center_2, R_ring,...
                                    V_water, V_media_1, V_media_2, V)
%    R1 and Center1 is larger circle

direction   = ((1:4096) / 4096) * 2 * pi;
direction_q = ((1:512) / 512) * 2 * pi;

wavefront_analytical = ones(1, length(direction_q));

direction_q(direction_q > pi) = direction_q(direction_q > pi) - 2 * pi;

direction_end = zeros(1, length(direction));
W = zeros(1, length(direction));
refracted = zeros(1, length(direction));

if norm([point_x, point_y] - Center_2) < R2
    for k = 1:length(direction)
        
        TOF = 0;
        [x1, y1, ~, l, angle_normal] = hitCircle(point_x, point_y, direction(k), Center_2(1), Center_2(2), R2, 1);
        TOF = TOF + l / V_media_2;
        direction1 = angle_normal + real(asin(sin(direction(k) - angle_normal)) * V_media_1 / V_media_2);

        [x2, y2, ~, l, angle_normal] = hitCircle(x1, y1, direction1, Center_1(1), Center_1(2), R1, 1);
        TOF = TOF + l / V_media_1;
        direction2 = angle_normal + real(asin(sin(direction1 - angle_normal)) * V_water / V_media_1);

        [x3, y3, ~, l, ~] = hitCircle(x2, y2, direction2, 0, 0, R_ring, 1);
        TOF = TOF + l / V_water;

%         figure(1);
%         hold off;
%         plot([point_x, x1, x2, x3], [point_y, y1, y2, y3]);
%         hold on;
%         scatter([point_x, x1, x2, x3], [point_y, y1, y2, y3]);
%         axis equal;
%         xlim([-5e-2, 5e-2]);
%         ylim([-5e-2, 5e-2]);
%         plot(R1 * cos(direction) + Center_1(1), R1 * sin(direction) + Center_1(2));
%         plot(R2 * cos(direction) + Center_2(1), R2 * sin(direction) + Center_2(2));
%         plot(R_ring * cos(direction), R_ring * sin(direction));
%         drawnow;

        direction_end(k) = angle(x3 - point_x + (y3 - point_y) * 1i);
        L = norm([x3 - point_x, y3 - point_y]);
        
        W(k) = L - TOF * V;        
    end
    
    wavefront_analytical = interp1([direction_end - 2 * pi, direction_end, direction_end + 2 * pi], [W, W, W], direction_q, 'nearest', 'extrap');
else
    for k = 1:length(direction)
        TOF = 0;
        [x1, y1, hit, l, angle_normal] = hitCircle(point_x, point_y, direction(k), Center_2(1), Center_2(2), R2, 0);
        if(hit)
            TOF = TOF + l / V_media_1;
            direction1 = angle_normal + real(asin(sin(direction(k) - angle_normal)) * V_media_2 / V_media_1);

            [x2, y2, ~, l, angle_normal] = hitCircle(x1, y1, direction1, Center_2(1), Center_2(2), R2, 1);
            TOF = TOF + l / V_media_2;
            direction2 = angle_normal + real(asin(sin(direction1 - angle_normal)) * V_media_1 / V_media_2);

            [x3, y3, ~, l, angle_normal] = hitCircle(x2, y2, direction2, Center_1(1), Center_1(2), R1, 1);
            TOF = TOF + l / V_media_1;
            direction3 = angle_normal + real(asin(sin(direction2 - angle_normal)) * V_water / V_media_1);

            [x4, y4, ~, l, ~] = hitCircle(x3, y3, direction3, 0, 0, R_ring, 1);
            TOF = TOF + l / V_water;
            
%             figure(1);
%             hold off;
%             plot([point_x, x1, x2, x3, x4], [point_y, y1, y2, y3, y4]);
%             hold on;
%             scatter([point_x, x1, x2, x3, x4], [point_y, y1, y2, y3, y4]);
%             axis equal;
%             xlim([-5e-2, 5e-2]);
%             ylim([-5e-2, 5e-2]);
%             plot(R1 * cos(direction) + Center_1(1), R1 * sin(direction) + Center_1(2));
%             plot(R2 * cos(direction) + Center_2(1), R2 * sin(direction) + Center_2(2));
%             plot(R_ring * cos(direction), R_ring * sin(direction));
%             plot(point_x + (10e-4 + W(refracted > 0) / 3) * 10 .* cos(direction_end(refracted > 0)), ...
%                  point_y + (10e-4 + W(refracted > 0) / 3) * 10 .* sin(direction_end(refracted > 0)), 'r');
%             plot(point_x + (10e-4 + W(refracted < 1) / 3) * 10 .* cos(direction_end(refracted < 1)), ...
%                  point_y + (10e-4 + W(refracted < 1) / 3) * 10 .* sin(direction_end(refracted < 1)), 'b');
%             drawnow;

            refracted(k) = 1;
            
            direction_end(k) = angle(x4 - point_x + (y4 - point_y) * 1i);
            L = norm([x4 - point_x, y4 - point_y]);
            W(k) = L - TOF * V;
        else
            [x1, y1, ~, l, angle_normal] = hitCircle(point_x, point_y, direction(k), Center_1(1), Center_1(2), R1, 1);
            TOF = TOF + l / V_media_1;
            direction1 = angle_normal + real(asin(sin(direction(k) - angle_normal)) * V_water / V_media_1);

            [x2, y2, ~, l, ~] = hitCircle(x1, y1, direction1, 0, 0, R_ring, 1);
            TOF = TOF + l / V_water;
            
%             figure(1);
%             hold off;
%             plot([point_x, x1, x2], [point_y, y1, y2]);
%             hold on;
%             scatter([point_x, x1, x2], [point_y, y1, y2]);
%             axis equal;
%             xlim([-5e-2, 5e-2]);
%             ylim([-5e-2, 5e-2]);
%             plot(R1 * cos(direction) + Center_1(1), R1 * sin(direction) + Center_1(2));
%             plot(R2 * cos(direction) + Center_2(1), R2 * sin(direction) + Center_2(2));
%             plot(R_ring * cos(direction), R_ring * sin(direction));
%             plot(point_x + (10e-4 + W(refracted > 0) / 3) * 10 .* cos(direction_end(refracted > 0)), ...
%                  point_y + (10e-4 + W(refracted > 0) / 3) * 10 .* sin(direction_end(refracted > 0)), 'r');
%             plot(point_x + (10e-4 + W(refracted < 1) / 3) * 10 .* cos(direction_end(refracted < 1)), ...
%                  point_y + (10e-4 + W(refracted < 1) / 3) * 10 .* sin(direction_end(refracted < 1)), 'b');
%             drawnow;
            
            direction_end(k) = angle(x2 - point_x + (y2 - point_y) * 1i);
            L = norm([x2 - point_x, y2 - point_y]);
            W(k) = L - TOF * V;
        end
    end
    
    for m = 1:length(direction_q)
       d_min1 = inf;
       d_min2 = inf;
       for n = 1:length(direction_end)
           d = mod(direction_q(m) - direction_end(n), 2 * pi);
           if(d > pi)
               d = 2 * pi - d;
           end
           if(refracted(n))
              if(d < d_min1)
                  d_min1 = d;
                  n1 = n;
              end
           else
              if(d < d_min2)
                  d_min2 = d;
                  n2 = n;
              end
           end
       end
       if(d_min1 > 0.01)
           wavefront_analytical(m) = W(n2);
       elseif(d_min2 > 0.01)
           wavefront_analytical(m) = W(n1);
       else
           wavefront_analytical(m) = max(W(n1), W(n2));
       end
    end
end
end






