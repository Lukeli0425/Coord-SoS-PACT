function WholeImageCorrected = JointAnImage(Look_y_id, Look_x_id, WholeImageCorrected, CorrectedImage, ImageX, ImageY)

    idxc = round(length(ImageX) / 2);
    idyc = round(length(ImageY) / 2);
    
%     CorrectedImage(1,:)   = CorrectedImage(1,:) * 2;
%     CorrectedImage(end,:) = CorrectedImage(end,:) * 2;
%     CorrectedImage(:,1)   = CorrectedImage(:,1) * 2;
%     CorrectedImage(:,end) = CorrectedImage(:,end) * 2;

    CorrectedImage_zeropad = zeros(size(WholeImageCorrected));
    CorrectedImage_zeropad(1:length(ImageY) - 1, 1:length(ImageX) - 1) = CorrectedImage(1:end - 1, 1:end - 1);
    CorrectedImage_zeropad = circshift(CorrectedImage_zeropad, Look_y_id - idyc, 1);
    CorrectedImage_zeropad = circshift(CorrectedImage_zeropad, Look_x_id - idxc, 2);
    WholeImageCorrected = WholeImageCorrected + CorrectedImage_zeropad;
end