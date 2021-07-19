clear
clc
close all

%data loading (here we use the AR dataset as an example)
load('usps.mat');

% -------------------------------------------------------------------------
% parameter setting
par.nClass    =   length(unique(tr_label)); % the number of classes in the subset of AR database
param = [];
param.lambda = 1e-4;
param.mu= 1e-2;

% data and labels for training and test samples
%--------------------------------------------------------------------------
Tr_DAT   =   double(tr_fea(:,tr_label<=par.nClass));
trls     =  tr_label(tr_label<=par.nClass);
Tt_DAT   =   double(ts_fea(:,ts_label<=par.nClass));
ttls     =   ts_label(ts_label<=par.nClass);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels


train_tol= size(Tr_DAT,2);
test_tol = size(Tt_DAT,2);
ClassNum = par.nClass;
%--------------------------------------------------------------------------
%eigenface extracting
[disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,35);
tr_dat  =  disc_set'*Tr_DAT;
tt_dat  =  disc_set'*Tt_DAT;
    
% normalize to unit L2 norm
tr_dat = normc(tr_dat);
tt_dat = normc(tt_dat);
ID = zeros(1,test_tol);
X = tr_dat;

% pre-computation
XTX = X'*X;
temp_X = pinv(XTX+(param.mu+2*param.lambda)/2*eye(size(XTX)));
   
for i=1:test_tol
     y = tt_dat(:,i);
     % coding
     [z,c] = ANCR(X, temp_X, y, param);
     % classification
     residual = ANCR_res(X,y,c,trls);
     [~,index]=min(residual);
     ID(i)=index;
end
 cornum     =   sum(ID==ttls');
 reg_rate=cornum/length(ttls); % recognition rate
 
disp('ANCR total accuracy rate is:')
 disp(reg_rate)
% output recognition result
fid=fopen('C:\Users\Administrator\Desktop\ANCR1\result_matlab.txt','w+');%写入文件路径
fprintf(fid,'%s','ANCR total accuracy rate is:  ');
fprintf(fid,'%f',reg_rate);
fclose(fid);
