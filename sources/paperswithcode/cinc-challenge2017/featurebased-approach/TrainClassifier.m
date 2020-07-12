function TrainClassifier(feature_file)
% This function extracts features for each record present  in a folder
%
%  Input:
%       - feature_file:         file containing table with extracted
%                               features in different records
%       
% --
% ECG classification from single-lead segments using Deep Convolutional Neural 
% Networks and Feature-Based Approaches - December 2017
% 
% Released under the GNU General Public License
%
% Copyright (C) 2017  Fernando Andreotti, Oliver Carr
% University of Oxford, Insitute of Biomedical Engineering, CIBIM Lab - Oxford 2017
% fernando.andreotti@eng.ox.ac.uk
%
% 
% For more information visit: https://github.com/fernandoandreotti/cinc-challenge2017
% 
% Referencing this work
%
% Andreotti, F., Carr, O., Pimentel, M.A.F., Mahdi, A., & De Vos, M. (2017). 
% Comparing Feature Based Classifiers and Convolutional Neural Networks to Detect 
% Arrhythmia from Short Segments of ECG. In Computing in Cardiology. Rennes (France).
%
% Last updated : December 2017
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

load(feature_file)
NFEAT=size(allfeats,2);
NFEAT=NFEAT-2;

% Get summary statistics on the distribution of the features in each
% signal, using following:
% - median
% - inter quartile range
% - range
% - min value
% - max value
% - 25% perctile
% - 50% perctile
% - 75% percentile
% - Real coefficients of Hilbert transform 
% - Absolute values of Hilbert transform
% - Skewness
% - Kurtosis
% 
feat = zeros(max(allfeats.rec_number),16*NFEAT);
for i=1:max(allfeats.rec_number)
    fprintf('Processing record %d .. \n',i)
    ind=find(table2array(allfeats(:,1))==i);
    feat(i,1:NFEAT)=nanmean(table2array(allfeats(ind,3:end)));
    feat(i,1*NFEAT+1:2*NFEAT)=nanstd(table2array(allfeats(ind,3:end)));
    if length(ind)>2
        PCAn=pca(table2array(allfeats(ind,3:end)));
        feat(i,2*NFEAT+1:3*NFEAT)=PCAn(:,1);
        feat(i,3*NFEAT+1:4*NFEAT)=PCAn(:,2);
    else
        feat(i,2*NFEAT+1:3*NFEAT)=NaN;
        feat(i,3*NFEAT+1:4*NFEAT)=NaN;
    end
    feat(i,4*NFEAT+1:5*NFEAT)=nanmedian(table2array(allfeats(ind,3:end)));
    feat(i,5*NFEAT+1:6*NFEAT)=iqr(table2array(allfeats(ind,3:end)));
    feat(i,6*NFEAT+1:7*NFEAT)=range(table2array(allfeats(ind,3:end)));
    feat(i,7*NFEAT+1:8*NFEAT)=min(table2array(allfeats(ind,3:end)));
    feat(i,8*NFEAT+1:9*NFEAT)=max(table2array(allfeats(ind,3:end)));
    feat(i,9*NFEAT+1:10*NFEAT)=prctile(table2array(allfeats(ind,3:end)),25);
    feat(i,10*NFEAT+1:11*NFEAT)=prctile(table2array(allfeats(ind,3:end)),50);
    feat(i,11*NFEAT+1:12*NFEAT)=prctile(table2array(allfeats(ind,3:end)),75);
    HIL=hilbert(table2array(allfeats(ind,3:end)));
    feat(i,12*NFEAT+1:13*NFEAT)=real(HIL(1,:));
    feat(i,13*NFEAT+1:14*NFEAT)=abs(HIL(1,:));
    feat(i,14*NFEAT+1:15*NFEAT)=skewness(table2array(allfeats(ind,3:end)));
    feat(i,15*NFEAT+1:16*NFEAT)=kurtosis(table2array(allfeats(ind,3:end))); 
end

In = feat;
Ntrain = size(In,1);
In(isnan(In)) = 0;
% Standardizing input
In = In - mean(In);
In = In./std(In);

labels = {'A' 'N' 'O' '~'};
Out = reference_tab{:,2};
Outbi = cell2mat(cellfun(@(x) strcmp(x,labels),Out,'UniformOutput',0));
Outde = bi2de(Outbi);
Outde(Outde == 4) = 3;
Outde(Outde == 8) = 4;
clear Out
rng(1); % For reproducibility
%% Perform cross-validation
%== Subset sampling
k = 5;
cv = cvpartition(Outde,'kfold',k);
confusion = zeros(4,4,k);
F1save = zeros(k,4);
F1_best = 0;
for i=1:k
    fprintf('Cross-validation loop %d \n',i)
    trainidx = find(training(cv,i));
    trainidx = trainidx(randperm(length(trainidx)));
    testidx  = find(test(cv,i));
    %% Bagged trees (oversampled)
    ens = fitensemble(In(trainidx,:),Outde(trainidx),'Bag',50,'Tree','type','classification');
    [~,probTree] = predict(ens,In(testidx,:));
    
    %% Neural networks
    net = patternnet(10);
    net = train(net,In(trainidx,:)',Outbi(trainidx,:)');            
    probNN = net(In(testidx,:)')';    
    
    %% Combining methods
    C = cat(3,probTree,probNN);
    C = mean(C,3);
    estimate = zeros(size(C,1),1);
    for r = 1:size(C,1)
        [~,estimate(r)] = max(C(r,:));
    end
    confmat = confusionmat(Outde(testidx),estimate);
    confusion(:,:,i) = confmat;    
    F1 = zeros(1,4);
    for j = 1:4
        F1(j)=2*confmat(j,j)/(sum(confmat(j,:))+sum(confmat(:,j)));
        fprintf('F1 measure for %s rhythm: %1.4f \n',labels{j},F1(j))
    end
    F1save(i,:) = F1;
    
    if F1 > F1_best
        F1_best = F1;
        ensTree_best = compact(ens);
        nnet_best = net;
    end
end
%% Producing statistics
confusion = sum(confusion,3);
F1 = zeros(1,4);
for i = 1:4
    F1(i)=2*confusion(i,i)/(sum(confusion(i,:))+sum(confusion(:,i)));
    fprintf('F1 measure for %s rhythm: %1.4f \n',labels{i},F1(i))
end
fprintf('Final F1 measure:  %1.4f\n',mean(F1))

%% Save output
save('results_allfeat.mat','F1save','F1_best')
save('ensTree.mat','ensTree_best')
save('nNets.mat','nnet_best')








