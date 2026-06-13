close all; clear all; clear variables; clear global; clc;    % clean desk
% load the file data for training the CNN
%train1 = imageDatastore('G:\PhD\Thesis Code in Process\Abhishek_Deep\outputfolder\03_CAT\A_Cat','IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames'); % use imageDatastore for loading the two image categories
train2 = imageDatastore('G:\PhD\Thesis Code in Process\Abhishek_Deep\outputfolder\03_CAT\F_Cat','IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
siz1=size(train2.Files);

for x1=1:siz1                         % print out how many images we have for each category
%     Y1 = readimage(train1,x1);                      % read one example image
%     Y2 = readimage(train2,x1);
%     Y3=imresize(Y1, [360 480]); 
%     Y4=imresize(Y2, [360 480]);
%      Y2=imresize(I2, [360 480]);
%      Y5=imsubtract(Y4,Y3);
%      Y3=Y1-Y2;
     [Y6]= im2bw(Y5,0.04);
     ms=Y6;
mso=ms(:,:,1)>0;
ho=imfill(mso,'holes');
se1=strel('square',3);
Y7=imdilate(ho,se1);

    folder3 = 'G:\PhD\Thesis Code in Process\Abhishek_Deep\outputfolder\03_CAT\L_Cat\';
    imwrite(Y7,fullfile(folder3,sprintf('GT_%d.png',x1)));
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
