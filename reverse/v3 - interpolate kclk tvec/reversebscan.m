%% Setup

clc; close all; clear;

% Load B-Scan
jpg_path = "/Users/belindaserafine/Documents/School/Capstone/Capstone Datasets/B-Scans/NEH_UT_2021RetinalOCTDataset/NORMAL/1/OS/000_normal.jpg";
bscan_name = "000_normal.jpg";
bscan = imread(jpg_path);
bscan = im2gray(bscan);
bscan = double(bscan).'; % transpose to match matrix to image orientation (so we dont have to tranpose the other parameter arrays)
[num_ascans, num_depth] = size(bscan); % [768x496]
signal_length = num_depth;

% Define sampling grid
tvec = (0:signal_length-1);
kclk = 0.05 * rand(1,signal_length) - 0.025;
kclk = tvec + kclk;
kclk(1) = 0;
kclk = round(kclk);

% Define OCT parameters for s4
maxmin      = [800e-9, 900e-9];
a_coeffs    = [0, -4 * 10^-11, 0];

% Define background to be subtracted at s6
mynoise = randn(1, signal_length); % gaussian/white noise with mean = 0, var = 0.01
mybg = ones(1, signal_length) * mean(bscan, 'all') + mynoise;
mybg = round(mybg);

% Storage
raw_signals  = complex(zeros(num_ascans, signal_length));
b_recon = complex(zeros(num_ascans, signal_length));

%% Forward pipeline

for i = 1:num_ascans
    ascan = bscan(i, :);                    % A single A-scan from image
    s1 = gs_map_r(ascan);                   % Undo greyscale mapping
    s2 = log_sc_r(s1);                      % Convert mag to log space
    s3 = fft_r(s2);                         % IFFT to depth domain
    s4 = disp_c_r(s3, maxmin, a_coeffs);    % Add phase dispersion
    s5 = k_lin_r(s4, tvec, kclk);           % Map to nonlinear k-clock
    s6 = bg_sub_r(s5, mybg);                % Add background
    raw_signals(i, :) = s6;
end

% Scale raw values to fit int32
max_val = max(abs(raw_signals(:)));
scale = double(intmax('int32')) / max_val;
raw_signals = raw_signals * scale;

% Round values
raw_signals_real = round(real(raw_signals));
raw_signals_im = round(imag(raw_signals));

%% Save parameters and outputs to MAT file

out_file = "forward_pipeline_outputs.mat";
save(out_file, "raw_signals", "bscan_name", "bscan", "tvec", "kclk", "maxmin", "a_coeffs", "mybg");

%% FUNCTIONS

%% undo 8-bit normalization/greyscale mapping
function output = gs_map_r(input)
    output = input / 255; % normalize to 0-1
    output = (output * (max(input) - min(input))) + min(input);
end


%% background subtraction
function output = bg_sub_r(input, bg)
    output = input + bg;
end


%% log scaling
function output = log_sc_r(input)
    mag = 10.^(input/20);
    output = mag + 0i;  % complex representation
end


%% fft
function output = fft_r(input)
    output = ifft(input);
end


%% dispersion correction
function output = disp_c_r(input, maxmin, a_coeffs)
    % add dispersion phase
    N = length(input);
    lambda = linspace(maxmin(1), maxmin(2), N);
    k = 2*pi ./ lambda;
    k0 = mean(k);
    phi = a_coeffs(2)*(k-k0).^2 + a_coeffs(3)*(k-k0).^3;
    output = input .* exp(1i * phi);
end


%% klinearization
function output = k_lin_r(linear_y, linear_x, nonlinear_x)
    % map from linear k-space to nonlinear k-clock
    output = interp1(linear_x, linear_y, nonlinear_x, 'pchip', 'extrap');
end
