close all; clear all; clear variables; clear global; clc;    % clean desk
% load the file data for training the CNN
train1 = imageDatastore('D:\DRIVES\F Drive\ImportPixelLabeledDatasetExample\Dataset\F_All','FileExtensions','.png','LabelSource','foldernames'); % use imageDatastore for loading the two image categories
train2 = imageDatastore('D:\DRIVES\F Drive\ImportPixelLabeledDatasetExample\Dataset\L_All','FileExtensions','.png','LabelSource','foldernames');
siz1=size(train1.Files);

for x1=1:siz1                         % print out how many images we have for each category
    Y1 = readimage(train1,x1);                      % read one example image
    Y2 = readimage(train2,x1);
    Y3=imresize(Y1, [360 480]); 
    Y4=imresize(Y2, [360 480]);
    Y5=imsubtract(Y3,Y4);
    [Y6]= im2bw(Y5,0.1);
    
    ms=Y6;
    mso=ms(:,:,1)>0;
    ho=imfill(mso,'holes');
    se1=strel('square',1);
    Y7=imdilate(ho,se1);
    folder3 = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2\LabelsReszed';
    imwrite(Y7,fullfile(folder3,sprintf('F_%d.png',x1)));

    folder4 = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2\ImagesReszed';
    imwrite(Y3,fullfile(folder4,sprintf('F_%d.png',x1)));

    folder5 = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2\Labels';
    imwrite(Y5,fullfile(folder5,sprintf('F_%d.png',x1)));

    folder6 = 'D:\claude\SemanticSegmentationUsingFCN-AlexNet1\Dataset2\Images';
    imwrite(Y3,fullfile(folder6,sprintf('F_%d.png',x1)));
end
% train2 = imageDatastore('C:\Users\abhi\Documents\MATLAB\Abhishek_Deep\fddf\All_SD_R_Forged\Aaa\SULFA\IM_Forg\11','IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
% siz2=size(train2.Files)
% 
% for x2=1:siz2                         % print out how many images we have for each category
%     Y3 = readimage(train2,x2);
%     Y4=imresize(Y3, [360 480]);
%     folder3 = 'C:\Users\abhi\Documents\MATLAB\Abhishek_Deep\fddf\All_SD_R_Forged\Aaa\SULFA\IM_Forg\11\';
%     imwrite(Y4,fullfile(folder3,sprintf('GT_%d.png',x2)));
% end
