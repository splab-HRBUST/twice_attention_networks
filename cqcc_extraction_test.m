clear; close all; clc;

% add required libraries to the path
addpath(genpath('baseline'));
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('GMM'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('tDCF_v1'));

% set here the experiment to run (access and feature type)
access_type = 'LA'; % LA for logical or PA for physical
feature_type = 'CQCC'; % LFCC or CQCC

% set paths to the wave files and protocols
% TODO: in this code we assume that the data follows the directory structure:
%
% ASVspoof_root/
%   |- data_physical
%      |- ASVspoof2019_LA_dev_asv_scores_v1.txt
% 	   |- ASVspoof2019_LA_dev_v1/
% 	   |- ASVspoof2019_LA_protocols_v1/
% 	   |- ASVspoof2019_LA_train_v1/
%   |- data_physical
%      |- ASVspoof2019_PA_dev_asv_scores_v1.txt
%      |- ASVspoof2019_PA_dev_v1/
%      |- ASVspoof2019_PA_protocols_v1/
%      |- ASVspoof2019_PA_train_v1/
% /home/chenchen/Data/SpeakerRecognition/ASVspoof/2019/LA/ASVspoof2019_LA_cm_protocols
% pathToASVspoof2019Data = '/path/to/ASVspoof_root/';





pathToASVspoof2019Data = '/mnt/g813_u6/2019';
if strcmp(access_type, 'LA')
    pathToDatabase = fullfile(pathToASVspoof2019Data, 'LA');
else
    pathToDatabase = fullfile(pathToASVspoof2019Data, 'PA');
end

trainProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_cm_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.eval.trl.txt'));
% /home/chenchen/Data/SpeakerRecognition/ASVspoof/2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
disp(['+++++++++++++++++++++++',trainProtocolFile]);
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filename = protocol{2};
sys_id = protocol{4};
key = protocol{5};
% get indices of genuine and spoof files
% bonafideIdx = find(strcmp(key,'bonafide'));
% spoofIdx = find(strcmp(key,'spoof'));

disp('Extracting features for training data...');
dset_name = 'eval';
cache_file_name = fullfile(['cache_' dset_name '_' access_type '_cqcc1.mat']);
disp(['cache_file_name=',cache_file_name]);
parfor i=1:length(filename)
    % extract CQCC feature
    filePath = fullfile(pathToDatabase,['ASVspoof2019_' access_type '_eval/flac'],[filename{i} '.flac']);
    [x,fs] = audioread(filePath);

    % padding to make data-length = 64000(4s)
    x_len = size(x,1);
    max_len = 64000;
    if x_len >= max_len
        x = x(1:max_len);
    else % need to pad
        num_repeats = floor(max_len / x_len)+1;
        x_repeat = repmat(x,num_repeats,1);
        x= x_repeat(1:max_len);
    end
    
    if strcmp(feature_type,'LFCC')
        [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
        trainFeatureCell{i} = [stat delta double_delta]';
    elseif strcmp(feature_type,'CQCC')
        trainFeatureCell{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    end
    % 转换系统id:sys_id
    % convert sys_id
    switch sys_id{i}
        case '-'
            sys_id{i} = 0;
        case 'A01'
            sys_id{i} = 1;
        case 'A02'
            sys_id{i} = 2;
        case 'A03'
            sys_id{i} = 3;
        case 'A04'
            sys_id{i} = 4;
        case 'A05'
            sys_id{i} = 5;
        case 'A06'
            sys_id{i} = 6;
        %For PA:
        case 'A07'
            sys_id{i} = 7;
        case 'A08'
            sys_id{i} = 8;
        case 'A09'
            sys_id{i} = 9;
        case 'A10'
            sys_id{i} = 10;
        case 'A11'
            sys_id{i} = 11;
        case 'A12'
            sys_id{i} = 12;
        case 'A13'
            sys_id{i} = 13;
        case 'A14'
            sys_id{i} = 14;
        case 'A15'
            sys_id{i} = 15;
        case 'A16'
            sys_id{i} = 16;
        case 'A17'
            sys_id{i} = 17;
        case 'A18'
            sys_id{i} = 18;
        case 'A19'
            sys_id{i} = 19;
        case 'AA'
            sys_id{i} = 20;
        case 'AB'
            sys_id{i} = 21;
        case 'AC'
            sys_id{i} = 22;
        case 'BA'
            sys_id{i} = 23;
        case 'BB'
            sys_id{i} = 24;
        case 'BC'
            sys_id{i} = 25;
        case 'CA'
            sys_id{i} = 26;
        case 'CB'
            sys_id{i} = 27;
        case 'CC'
            sys_id{i} = 28;
        otherwise
            disp('error converting system id!')
    end
end

% convert key
data_y = strcmp(key,'bonafide');
data_x = transpose(trainFeatureCell);
save(cache_file_name, 'filename', 'data_x', 'data_y', 'sys_id', '-v7.3')
disp('finished!')
