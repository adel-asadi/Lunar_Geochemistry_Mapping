%This code generates the Iron and Titanium abundance maps of the lunar
%surface with a hybrid of machine learning algorithms, based on the 5-band
%multi-spectral data of Clementine's Ultraviolet/Visible provided by USGS.

%The code requires a very high amount of RAM (more than 200 GB) and CPU to run; thus, please
%use a very powerful computer to apply this method.

%Please download the whole TIFF image from the website of US Geologcal
%Survey in order to map the whole lunar surface after building the
%regression model, via this link and move it to MATLAB directory folder:
%https://planetarymaps.usgs.gov/mosaic/Lunar_Clementine_UVVIS_WarpMosaic_5Bands_200m.tif

%The first step is to generate synthetic data using Synthetic Minority
%Oversampling Technique (SMOTE Algorithm) via the available 47 real data of
%the samples derived by Apollo and Luna missions (available in "lunar.m"
%script:

lunar;
lunar_inputs=lunar_data(:,1:5);
lunar_fe=lunar_data(:,6);
lunar_ti=lunar_data(:,7);
smote_input_fe=[lunar_inputs lunar_fe];
smote_input_ti=[lunar_inputs lunar_ti];
N=1000; %Percentage of Data Augmentation.
k=5; %Number of neighbors for each point.
fe_smote = mySMOTE(smote_input_fe, N, k);
ti_smote = mySMOTE(smote_input_ti, N, k);

%Removing the original data from the set to bes used just for testing:

fe_smote=fe_smote(48:end,:);
ti_smote=ti_smote(48:end,:);

%Training the Gaussian Process Regression algorithm (GPR) for Iron Oxide
%and Titanium Oxide prediction, with Rational Quadratic and Matern 5/2
%Kernel Functions, respectively. For TiO2 regression modeling, we are using
%natural logarithms of the original value in order to avoid negative value
%generation by GPR model.

[trainedModelFe, FevalidationRMSE] = trainRegressionModelFe(fe_smote);
ti_smote(:,end)=log(ti_smote(:,end));
[trainedModelTi, TivalidationRMSE] = trainRegressionModelTi(ti_smote);

%Checking the accuracy of the model on testing data:

feo;
yfit_fe=trainedModelFe.predictFcn(lunar_inputs);
fe_corr=corrcoef(fe,yfit_fe);
fe_mse=immse(fe,yfit_fe);
fe_mae=mean(abs(fe-yfit_fe));
tio2;
yfit_ti=trainedModelTi.predictFcn(lunar_inputs);
yfit_ti=exp(yfit_ti);
ti_corr=corrcoef(ti,yfit_ti);
ti_mse=immse(ti,yfit_ti);
ti_mae=mean(abs(ti-yfit_ti));

%Reading the 5-band image file in MATLAB:

tifimage = imread('Lunar_Clementine_UVVIS_WarpMosaic_5Bands_200m.tif');

%Extracting the matrix of Fractional Reflectance:
%The matrix will convert from 3D 16-bit to 2D double.

n=27273*54545;
fr_table=zeros(n,5);
s=0;
for i=1:27273
    for j=1:54545
        s=s+1;
        hh=double((squeeze(tifimage(i,j,:)))');
        fr_table(s,:)=(1.3700e-04)*hh;
    end
end

%Estimation of the Titanium Oxide weight on lunar surface:

nn=1:n;
ti_p=trainedModelTi.predictFcn(fr_table(nn,:));
ti_p=exp(ti_p);

%Generation of the 2D matrix of TiO2 abundance across lunar surface:

ti_map=reshape(ti_p,[54545,27273]);
ti_map=ti_map';

%Generating the full map of TiO2:

imagesc([-180,180],[-90,90],ti_map)
colormap('jet')
caxis([0 5])
c = colorbar('southoutside');
c.Label.String = 'Titanium Oxide Weight %';
xlabel('Longitude')
ylabel('Latitude')
title('Lunar Surface TiO2 Abundance Map')

%Estimation of the Iron Oxide weight on lunar surface:

fe_p=trainedModelFe.predictFcn(fr_table(nn,:));

%Generation of the 2D matrix of FeO abundance across lunar surface:

fe_map=reshape(fe_p,[54545,27273]);
fe_map=fe_map';

%Generating the full map of FeO:

imagesc([-180,180],[-90,90],fe_map)
colormap('jet')
caxis([0 10])
c = colorbar('southoutside');
c.Label.String = 'Iron Oxide Weight %';
xlabel('Longitude')
ylabel('Latitude')
title('Lunar Surface FeO Abundance Map')

%The End
