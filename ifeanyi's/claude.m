clc; close all; clear;

%% LOAD IMAGE + DEFINE LENGTH

img_path = "000_normal.jpg";
img = imread(img_path);
bscan = double(img(:, :, 1));   % use one channel
[num_depth, numascans] = size(bscan);
signal_length = num_depth;

%% DEFINE SAMPLING GRIDS

tvec = (0:signal_length-1)';
kclk = 0.05 * rand(signal_length,1) - 0.025;
kclk = tvec + kclk;
kclk(1) = 0;

%% OCT PARAMETERS
maxmin      = [800e-9, 900e-9];
a_coeffs    = [0, 0.001, 0.003];

%% STORAGE
og_sig  = complex(zeros(signal_length, numascans));
b_recon = complex(zeros(signal_length, numascans));
mybg = zeros(signal_length,1);


%% REVERSE PIPELINE (simulate raw from processed image)

for i = 1:numascans
    ascan = bscan(:, i);                    % Real data from image
    s1 = log_sc_r(ascan);                   % Convert mag to log space
    s2 = fft_r(s1);                         % IFFT to depth domain
    s3 = disp_c_r(s2, maxmin, a_coeffs);   % Add dispersion
    s4 = k_lin_r(s3, tvec, kclk);           % Map to nonlinear k-clock
    s5 = bg_sub_r(s4, mybg);                % Add background
    og_sig(:, i) = s5;
end


%% FORWARD PIPELINE (reconstruct)

for i = 1:numascans
    raw = og_sig(:, i);
    s1 = bg_sub_f(raw, mybg);               % Remove background
    s2 = k_lin_f(s1, tvec, kclk);           % Linearize k-space
    s3 = disp_c_f(s2, maxmin, a_coeffs);   % Correct dispersion
    s4 = fft_f(s3);                         % FFT to spatial domain
    s5 = log_sc_f(s4);                      % Log scale magnitude
    b_recon(:, i) = s5;
end


%% DISPLAY RESULT

figure(1)
imagesc(abs(b_recon))
colormap(gray)
title("Claude Reconstructed B-scan")

figure(2)
imagesc(abs(bscan))
colormap(gray)
title("Original B-scan")

figure(3)
imagesc(abs(b_recon - bscan))
colormap(hot)
title("Absolute Difference")
colorbar


%% ERROR METRIC

err = norm(abs(b_recon(:)) - abs(bscan(:))) / norm(abs(bscan(:)));
disp("Relative reconstruction error:")
disp(err)

%% ================== FUNCTIONS =============================


%% -------- BACKGROUND SUBTRACTION (SIMPLE + REVERSIBLE) --------
function output = bg_sub_f(input, bg)
    output = input - bg;
end

function output = bg_sub_r(input, bg)
    output = input + bg;
end

%% -------- DISPERSION CORRECTION (COMPLEX + REVERSIBLE) --------
function output = disp_c_f(input, maxmin, a_coeffs)
    % Forward: Remove dispersion (multiply by opposite phase)
    N = length(input);
    lambda = linspace(maxmin(1), maxmin(2), N)';
    k = 2*pi ./ lambda;
    k0 = mean(k);
    phi = a_coeffs(2)*(k-k0).^2 + a_coeffs(3)*(k-k0).^3;
    output = input .* exp(-1i * phi);
end

function output = disp_c_r(input, maxmin, a_coeffs)
    % Reverse: Add dispersion (multiply by phase)
    N = length(input);
    lambda = linspace(maxmin(1), maxmin(2), N)';
    k = 2*pi ./ lambda;
    k0 = mean(k);
    phi = a_coeffs(2)*(k-k0).^2 + a_coeffs(3)*(k-k0).^3;
    output = input .* exp(1i * phi);
end

%% -------- K-LINEARIZATION (INTERPOLATION) --------
function output = k_lin_f(nonlinear_y, linear_x, nonlinear_x)
    % Forward: Map from nonlinear k-clock to linear k-space
    assert(length(nonlinear_y) == length(nonlinear_x), ...
        'k_lin_f: signal and source grid must match');
    output = interp1(nonlinear_x, nonlinear_y, linear_x, 'pchip', 'extrap');
end

function output = k_lin_r(linear_y, linear_x, nonlinear_x)
    % Reverse: Map from linear k-space to nonlinear k-clock
    assert(length(linear_y) == length(linear_x), ...
        'k_lin_r: signal and source grid must match');
    output = interp1(linear_x, linear_y, nonlinear_x, 'pchip', 'extrap');
end

%% -------- FFT (FULL COMPLEX) --------
function output = fft_f(input)
    % Forward: FFT to spatial domain
    output = fft(input);
end

function output = fft_r(input)
    % Reverse: IFFT to k-space domain
    output = ifft(input);
end

%% -------- LOG SCALING (MAGNITUDE ONLY) --------
function output = log_sc_f(input)
    % Forward: Convert magnitude to dB scale
    % Input: complex data from FFT
    % Output: real dB magnitude (magnitude domain)
    epsv = 1e-12;
    output = 20*log10(abs(input) + epsv);
end

function output = log_sc_r(input)
    % Reverse: Convert from linear magnitude to complex
    % Input: real magnitude values (from image)
    % Output: complex data (assume zero phase initially)
    % 
    % Note: If input is already in dB, convert first:
    % mag = 10.^(input/20);
    % If input is linear magnitude:
    mag = input;
    output = mag + 0i;  % Complex representation with zero phase
end
