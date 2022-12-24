function [Look_x, Look_y, Look_x_id, Look_y_id] = FormPatchCenters(WholeImageX, WholeImageY, Mask, ImageX, ImageY)

    Move_step_x = round(length(ImageX) / 4);
    Move_step_y = round(length(ImageY) / 4);
    

    Look_x_id = 1:Move_step_x:length(WholeImageX);
    Look_y_id = 1:Move_step_y:length(WholeImageY);

    in_Mask = Mask(Look_y_id, Look_x_id);

    Look_x = WholeImageX(1:Move_step_x:end);
    Look_y = WholeImageY(1:Move_step_y:end);

    [Look_x_id, Look_y_id] = meshgrid(Look_x_id, Look_y_id);
    [Look_x   , Look_y]    = meshgrid(Look_x, Look_y);

    Look_x_id = Look_x_id(in_Mask);
    Look_y_id = Look_y_id(in_Mask);

    Look_x    = Look_x(in_Mask);
    Look_y    = Look_y(in_Mask);

    Look_x_id = reshape(Look_x_id, [], 1);
    Look_y_id = reshape(Look_y_id, [], 1);

    Look_x = reshape(Look_x, [], 1);
    Look_y = reshape(Look_y, [], 1);
end