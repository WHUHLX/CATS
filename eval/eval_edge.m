% % Command to run.
% % (echo "data_dir = '../output/epoch-x-test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
%
% % Data directory data_dir should be defined outsidedd
data_dir = './output/bsds/train/'

addpath(genpath('./toolbox.badacost.public'));
addpath(genpath('./edges'));

% Section 1: NMS process (formerly nms_process.m from HED repo).
disp('NMS process...')
mat_dir = fullfile(data_dir, 'mat');
nms_dir = fullfile(data_dir, 'nms');
mkdir(nms_dir)

files = dir(mat_dir);
files = files(3:end,:);  % It means all files except ./.. are considered.
mat_names = cell(1,size(files, 1));
nms_names = cell(1,size(files, 1));
for i = 1:size(files, 1),
    mat_names{i} = files(i).name;
    nms_names{i} = [files(i).name(1:end-4), '.png']; % Output PNG files.
end

for i = 1:size(mat_names,2),
    matObj = matfile(fullfile(mat_dir, mat_names{i})); % Read MAT files.
    varlist = who(matObj);
    x = matObj.(char(varlist));
    E=convTri(single(x),1);
    [Ox,Oy]=gradient2(convTri(E,4));
    [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
    O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);

    % BSDS500 use edgesNmsMex(E,O,1,4,1.01,8), while NYUDv2 use edgesNmsMex(E,O,4,4,1.01,8)
    E=edgesNmsMex(E,O,1,4,1.01,8);
    imwrite(uint8(E*255),fullfile(nms_dir, nms_names{i}))
end

% Section 2: Evaluate the edges (formerly EvalEdge.m from HED repo).
disp('Evaluate the edges...');


% gtDir  = './NYUD/mat';
gtDir  = './BSDS/test';
resDir = fullfile(data_dir, 'nms');

% Evaluation for BSDS500 dataset
edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.0075, 'thrs',99);

% Evaluation for NYUD dataset
% edgesEvalDir('resDir',resDir,'gtDir',gtDir, 'thin', 1, 'pDistr',{{'type','parfor'}},'maxDist',0.011, 'thrs',99);
%
figure; edgesEvalPlot(resDir,'CATS');
