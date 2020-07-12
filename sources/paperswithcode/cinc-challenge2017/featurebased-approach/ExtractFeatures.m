function ExtractFeatures(dbpath,varargin)
% This function extracts features for each record present  in a folder
%
%  Input:
%       - dbpath:         directory where database is
%       (optional inputs)
%           - useSegments:       segment signals into windows (bool)?
%           - windowSize:        size of window used in segmenting record
%           - percentageOverlap: overlap between windows
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


% Default arguments
slashchar = char('/'*isunix + '\'*(~isunix));
if ~strcmp(dbpath(end),slashchar)
    dbpath = [dbpath slashchar];
end
optargs = {1 10 0.8};  % default values for input arguments
newVals = cellfun(@(x) ~isempty(x), varargin);
optargs(newVals) = varargin(newVals);
[useSegments, windowSize, percentageOverlap] = optargs{:};
clear optargs newVals

% Parameters
NFEAT = 169; % number of features used
NFEAT_hrv = 113;

fs = 300;       % sampling frequency [Hz]

% Add subfunctions to matlab path
mainpath = (strrep(which(mfilename),['preparation' slashchar mfilename '.m'],''));
addpath(genpath([mainpath(1:end-length(mfilename)-2) 'subfunctions' slashchar])) % add subfunctions folder to path


% Find recordings
mkdir([dbpath 'featextract'])
cd([dbpath 'featextract' slashchar])
disp('Loading reference from Physionet..')
ref_filename = [dbpath 'REFERENCE.csv'];
websave(ref_filename,'https://physionet.org/challenge/2017/REFERENCE-v3.csv');
reference_tab = readtable(ref_filename,'ReadVariableNames',false);
fls = reference_tab{:,1};
clear dataArray delimiter ref_filename formatSpec fileID


%% Initialize loop
% Wide BP
Fhigh = 5;  % highpass frequency [Hz]
Flow = 45;   % low pass frequency [Hz]
Nbut = 10;     % order of Butterworth filter
d_bp= design(fdesign.bandpass('N,F3dB1,F3dB2',Nbut,Fhigh,Flow,fs),'butter');
[b_bp,a_bp] = tf(d_bp);

% Narrow BP
Fhigh = 1;  % highpass frequency [Hz]
Flow = 100;   % low pass frequency [Hz]
Nbut = 10;     % order of Butterworth filter
d_bp= design(fdesign.bandpass('N,F3dB1,F3dB2',Nbut,Fhigh,Flow,fs),'butter');
[b_bp2,a_bp2] = tf(d_bp);
clear Fhigh Flow Nbut d_bp

%% Run through files
allfeats = cell2table(cell(0,NFEAT+2));
for f = 1:length(fls)
    %% Loading data
    data = load([dbpath fls{f} '.mat']);
    fname = fls{f};
    signal = data.val;
    if size(signal,1)<size(signal,2), signal = signal'; end % make sure it's column vector
    signalraw =  signal;
    
    %% Preprocessing
    signal = filtfilt(b_bp,a_bp,signal);             % filtering narrow
    signal = detrend(signal);                        % detrending (optional)
    signal = signal - mean(signal);
    signal = signal/std(signal);                     % standardizing
    signalraw = filtfilt(b_bp2,a_bp2,signalraw);     % filtering wide
    signalraw = detrend(signalraw);                  % detrending (optional)
    signalraw = signalraw - mean(signalraw);
    signalraw = signalraw/std(signalraw);        % standardizing
    disp(['Preprocessed ' fname '...'])
    
    % Figuring out if segmentation is used
    if useSegments==1
        WINSIZE = windowSize; % window size (in sec)
        OLAP = percentageOverlap;
    else
        WINSIZE = length(signal)/fs;
        OLAP=0;
    end
    startp = 1;
    endp = WINSIZE*fs;
    looptrue = true;
    nseg = 1;
    while looptrue
        % Conditions to stop loop
        if length(signal) < WINSIZE*fs
            endp = length(signal);
            looptrue = false;
            continue
        end
        if nseg > 1
            startp(nseg) = startp(nseg-1) + round((1-OLAP)*WINSIZE*fs);
            if length(signal) - endp(nseg-1) < 0.5*WINSIZE*fs
                endp(nseg) = length(signal);
            else
                endp(nseg) = startp(nseg) + WINSIZE*fs -1;
            end
        end
        if endp(nseg) == length(signal)
            looptrue = false;
            nseg = nseg - 1;
        end
        nseg = nseg + 1;
    end
    
    % Obtain features for each available segment
    fetbag = {};
    feat_hrv = [];
    parfor n = 1:nseg
        % Get signal of interest
        sig_seg = signal(startp(n):endp(n));
        sig_segraw = signalraw(startp(n):endp(n));
        
        % QRS detect
        [qrsseg,featqrs] = multi_qrsdetect(sig_seg,fs,[fname '_s' num2str(n)]);
        
        % HRV features
        if length(qrsseg{end})>5 % if too few detections, returns zeros
            try
                feat_basic=HRV_features(sig_seg,qrsseg{end}./fs,fs);
                feats_poincare = get_poincare(qrsseg{end}./fs,fs);
                feat_hrv = [feat_basic, feats_poincare];
                feat_hrv(~isreal(feat_hrv)|isnan(feat_hrv)|isinf(feat_hrv)) = 0; % removing not numbers
            catch
                warning('Some HRV code failed.')
                feat_hrv = zeros(1,NFEAT_hrv);
            end
        else
            disp('Skipping HRV analysis due to shortage of peaks..')
            feat_hrv = zeros(1,NFEAT_hrv);
        end
        
        % Heart Rate features
        HRbpm = median(60./(diff(qrsseg{end}./fs)));
        %obvious cases: tachycardia ( > 100 beats per minute (bpm) in adults)
        feat_tachy = normcdf(HRbpm,120,20); % sampling from normal CDF
        %See e.g.   x = 10:10:200; p = normcdf(x,120,20); plot(x,p)
        
        %obvious cases: bradycardia ( < 60 bpm in adults)
        feat_brady = 1-normcdf(HRbpm,60,20);
        
        % SQI metrics
        feats_sqi = ecgsqi(sig_seg,qrsseg,fs);
        
        % Features on residual
        featsres = residualfeats(sig_segraw,fs,qrsseg{end});
        
        % Morphological features
        feats_morph = morphofeatures(sig_segraw,fs,qrsseg,[fname '_s' num2str(n)]);
        
        
        feat_fer=[featqrs,feat_tachy,feat_brady,double(feats_sqi),featsres,feats_morph];
        feat_fer(~isreal(feat_fer)|isnan(feat_fer)|isinf(feat_fer)) = 0; % removing not numbers
        
        % Save features to table for training
        feats = [feat_hrv,feat_fer];
        fetbag{n} = feats;
    end
    thisfeats = array2table([repmat(f,nseg,1),[1:nseg]',cell2mat(fetbag')]);%#ok<NBRAK>
    allfeats = [allfeats;thisfeats];
    
end
delete('gqrsdet*.*')
% diary off
%% Saving Output
% hardcoded feature names
names = {'rec_number' 'seg_number' 'sample_AFEv' 'meanRR' 'medianRR' 'SDNN' 'RMSSD' 'SDSD' 'NN50' 'pNN50' 'LFpeak' 'HFpeak' 'totalpower' 'LFpower' ...
    'HFpower' 'nLF' 'nHF' 'LFHF' 'PoincareSD1' 'PoincareSD2' 'SampEn' 'ApEn'  ...
    'RR' 'DET' 'ENTR' 'L' 'TKEO1'  'DAFa2' 'LZ' ...
    'Clvl1' 'Clvl2' 'Clvl3' 'Clvl4' 'Clvl5' 'Clvl6' 'Clvl7' 'Clvl8' 'Clvl9' ...
    'Clvl10' 'Dlvl1' 'Dlvl2' 'Dlvl3' 'Dlvl4' ...
    'Dlvl5' 'Dlvl6' 'Dlvl7' 'Dlvl8' 'Dlvl9' 'Dlvl10' ...
    'percR50' 'percR100' 'percR200' 'percR300' 'medRR' 'meddRR' 'iqrRR' 'iqrdRR' 'bins1' 'bins2' 'bins1nL' 'bins2nL' 'bins1nS' 'bins2nS' ...
    'edgebins1' 'edgebins2' 'edgebins1nL' 'edgebins2nL' 'edgebins1nS' 'edgebins2nS' 'minArea' 'minAreanL' 'minAreanS' ...
    'minCArea' 'minCAreanL' 'minCAreanS' 'Perim' 'PerimnL' 'PerimnS' 'PerimC' 'PerimCnL' 'PerimCnS' ...
    'DistCen' 'DistCennL' 'DistCennS' 'DistNN' 'DistNNnL' 'DistNNnS' 'DistNext' 'DistNextnL' 'DistNextnS' 'ClustDistMax' 'ClustDistMin' ...
    'ClustDistMean' 'ClustDistSTD' 'ClustDistMed' 'MajorAxis' 'percR3' 'percR5' 'percR10' 'percR20' 'percR30' 'percR40' ...
    'Xcent' 'Ycent' 'rad1' 'rad2' 'rad1rad2' 'theta' 'NoClust1' 'NoClust2' 'NoClust3' 'NoClust4' 'NoClust5' 'NoClust6' 'NoClust7'};
names = [names 'amp_varsqi' 'amp_stdsqi' 'amp_mean'];
names = [names 'tachy' 'brady' 'stdsqi' 'ksqi' 'ssqi' 'psqi'];
combs = nchoosek(1:5,2);
combs = num2cell(combs,2);
names = [names cellfun(@(x) strtrim(['bsqi_',num2str(x(1)) num2str(x(2))]),combs,'UniformOutput',false)'];
names = [names arrayfun(@(x) strtrim(['rsqi_',num2str(x)]),1:5,'UniformOutput',false)];
names = [names arrayfun(@(x) strtrim(['csqi_',num2str(x)]),1:5,'UniformOutput',false)];
names = [names arrayfun(@(x) strtrim(['xsqi_',num2str(x)]),1:5,'UniformOutput',false)];
names = [names 'res1' 'res2' 'res3' 'res4' 'res5'];
names = [names 'QRSheight','QRSwidth','QRSpow','noPwave','Pheight','Pwidth','Ppow','Theight',...
    'Twidth','Tpow','Theightnorm','Pheightnorm','Prelpow','PTrelpow','Trelpow','QTlen','PRlen'];
allfeats.Properties.VariableNames = names;
save(['allfeatures_olap' num2str(OLAP) '_win' num2str(WINSIZE) '.mat'],'allfeats','reference_tab');

