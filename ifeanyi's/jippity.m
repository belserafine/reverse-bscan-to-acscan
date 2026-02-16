
%% LOAD IMAGE + DEFINE LENGTH


img_path = "000_normal.jpg";
img = imread(img_path);

bscan = double(img(:, :, 1));   % use one channel
[num_depth, numascans] = size(bscan);

signal_length = num_depth;      % ← REQUIRED CHANGE


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

mybg = zeros(signal_length,1);  % safe placeholder

%% REVERSE PIPELINE (simulate raw)

for i = 1:numascans

    ascan = complex(bscan(:, i));

    s1 = log_sc_r(ascan);
    s2 = fft_r(s1);
    s3 = disp_c_r(s2, maxmin, a_coeffs);
    s4 = k_lin_r(s3, tvec, kclk);
    s5 = bg_sub_r(s4, mybg);

    og_sig(:, i) = s5;

end

%% FORWARD PIPELINE (reconstruct)

for i = 1:numascans

    raw = og_sig(:, i);

    s1 = bg_sub_f(raw, mybg);
    s2 = k_lin_f(s1, tvec, kclk);
    s3 = disp_c_f(s2, maxmin, a_coeffs);
    s4 = fft_f(s3);
    s5 = log_sc_f(s4);

    b_recon(:, i) = s5;

end


%% DISPLAY RESULT


figure(10)
imagesc(abs(b_recon))
colormap(gray)
title("GPT Reconstructed B-scan")


%% ERROR METRIC


err = norm(abs(b_recon(:)) - abs(bscan(:))) / norm(abs(bscan(:)));
disp("Relative reconstruction error:")
disp(err)

%% ================== FUNCTIONS =============================

function output = bg_sub_f(input, bg)
    output = input - bg;
end

function output = bg_sub_r(input, bg)
    output = input + bg;
end

%% -------- DISPERSION (COMPLEX + REVERSIBLE) --------

function output = disp_c_f(input, maxmin, a_coeffs)

    N = length(input);
    lambda = linspace(maxmin(1), maxmin(2), N)';
    k = 2*pi ./ lambda;
    k0 = mean(k);

    phi = a_coeffs(2)*(k-k0).^2 + a_coeffs(3)*(k-k0).^3;
    output = input .* exp(-1i * phi);

end

function output = disp_c_r(input, maxmin, a_coeffs)

    N = length(input);
    lambda = linspace(maxmin(1), maxmin(2), N)';
    k = 2*pi ./ lambda;
    k0 = mean(k);

    phi = a_coeffs(2)*(k-k0).^2 + a_coeffs(3)*(k-k0).^3;
    output = input .* exp(1i * phi);

end

%% -------- K-LINEARIZATION (SAFE + CONSISTENT) --------

function output = k_lin_f(bad_y, good_x, bad_x)

    assert(length(bad_y) == length(bad_x), ...
        "k_lin_f: signal and source grid must match");

    output = interp1(bad_x, bad_y, good_x, 'pchip', 'extrap');

end

function output = k_lin_r(good_y, good_x, bad_x)

    assert(length(good_y) == length(good_x), ...
        "k_lin_r: signal and source grid must match");

    output = interp1(good_x, good_y, bad_x, 'pchip', 'extrap');

end

%% -------- FFT (FULL COMPLEX) --------

function output = fft_f(input)
    output = fft(input);
end

function output = fft_r(input)
    output = ifft(input);
end

%% -------- REVERSIBLE LOG MODEL --------

function output = log_sc_f(input)
    epsv = 1e-12;
    output = 20*log10(abs(input) + epsv) .* exp(1i * angle(input));
end

function output = log_sc_r(input)
    mag = 10.^(real(input)/20);
    output = mag .* exp(1i * angle(input));
end
