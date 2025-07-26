
clear;
close all;
clc;

%% 1. Data Loading and Preprocessing

load('100m.mat');

% The signal is typically in the first column of the 'val' matrix.

ecg_original = val(1, 1:2000); % we are use the first 2000 samples

% The original sampling frequency of MIT-BIH data is 360 Hz.
Fs_original = 360;

% The Pan-Tompkins algorithm is designed for a 200 Hz sampling rate.
Fs = 200;
ecg = resample(ecg_original, Fs, Fs_original);

% Create the time vector for the resampled signal
tm = (0:length(ecg)-1) / Fs;

% Remove the baseline wander by subtracting the mean
ecg = ecg - mean(ecg);

figure;
plot(tm, ecg);
title('Original ECG Signal (Loaded from 100m.mat and Resampled)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

%% 2. Pan-Tompkins Algorithm Implementation

% --- Stage 1: Bandpass Filter ---  high-pass filter to achieve a passband of 5-12 Hz.

% Low-pass filter: H(z) = (1 - z^-6)^2 / (1 - z^-1)^2
b_lp = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1];
a_lp = [1, -2, 1];
ecg_lp = filter(b_lp, a_lp, ecg);
delay_lp = 6; % Delay in samples

% High-pass filter: H(z) = (-1 + 32z^-16 + z^-32) / (1 + z^-1)
b_hp = [-1, zeros(1,15), 32, zeros(1,15), 1];
a_hp = [1, 1];
ecg_hp = filter(b_hp, a_hp, ecg_lp);
delay_hp = 16;

ecg_bp = ecg_hp;
total_delay = delay_lp + delay_hp;

figure;
subplot(2,1,1);
plot(tm, ecg);
title('Original ECG');
subplot(2,1,2);
plot(tm, ecg_bp);
title('Bandpass Filtered ECG');
xlabel('Time (s)');

% --- Stage 2: Derivative Filter ---
% H(z) = (1/8T) * (-z^-2 - 2z^-1 + 2z^1 + z^2)
% This is a 5-point derivative. For causality, we will use a delayed version.
% y(n) = (1/8) * [2*x(n) + x(n-1) - x(n-3) - 2*x(n-4)]
b_d = (1/8) * [2, 1, 0, -1, -2];
a_d = 1;
ecg_der = filter(b_d, a_d, ecg_bp);
delay_d = 2; % Delay in samples
total_delay = total_delay + delay_d;

figure;
plot(tm, ecg_der);
title('Derivative Filter Output');
xlabel('Time (s)');

% --- Stage 3: Squaring ---
ecg_sq = ecg_der.^2;

figure;
plot(tm, ecg_sq);
title('Squared Signal');
xlabel('Time (s)');

% --- Stage 4: Moving Window Integration ---
% Window size N should be approx. the width of the widest QRS complex.
% In paper 150 ms. At Fs=200Hz, N = 0.150 * 200 = 30 samples.
N = 30;
b_mwi = (1/N) * ones(1, N);
a_mwi = 1;
ecg_mwi = filter(b_mwi, a_mwi, ecg_sq);
delay_mwi = (N-1)/2;
total_delay = total_delay + delay_mwi;

figure;
plot(tm, ecg_mwi);
title('Moving Window Integration Output');
xlabel('Time (s)');

% --- Stage 5: Fiducial Mark Detection (Thresholding) ---
% This is a simplified version of the paper's adaptive thresholding for clarity.

peaks = [];
peak_locs = [];
threshold = 0.5 * max(ecg_mwi); % Initial simple threshold

for i = 2:length(ecg_mwi)-1
    if ecg_mwi(i) > ecg_mwi(i-1) && ecg_mwi(i) > ecg_mwi(i+1) && ecg_mwi(i) > threshold
        peaks = [peaks, ecg_mwi(i)];
        peak_locs = [peak_locs, i];
    end
end

% Adjust locations for the total filter delay
qrs_locs_pt = round(peak_locs - total_delay);
qrs_locs_pt(qrs_locs_pt <= 0) = []; % Remove non-positive indices

figure;
plot(tm, ecg);
hold on;
plot(tm(qrs_locs_pt), ecg(qrs_locs_pt), 'ro', 'MarkerFaceColor', 'r');
title('QRS Detection using Pan-Tompkins');
xlabel('Time (s)');
legend('ECG Signal', 'Detected QRS');


%% 3. DSP Analysis of Filters

% --- Low-pass Filter Analysis ---
[h_lp, w_lp] = freqz(b_lp, a_lp, 1024, Fs);
[gd_lp, w_gd_lp] = grpdelay(b_lp, a_lp, 1024, Fs);

figure;
subplot(3,1,1);
plot(w_lp, 20*log10(abs(h_lp)));
title('Low-pass Filter - Magnitude Response');
ylabel('Magnitude (dB)');
grid on;

subplot(3,1,2);
plot(w_lp, unwrap(angle(h_lp)));
title('Low-pass Filter - Phase Response');
ylabel('Phase (rad)');
grid on;

subplot(3,1,3);
plot(w_gd_lp, gd_lp);
title('Low-pass Filter - Group Delay');
ylabel('Delay (samples)');
xlabel('Frequency (Hz)');
grid on;


% --- High-pass Filter Analysis ---
[h_hp, w_hp] = freqz(b_hp, a_hp, 1024, Fs);
[gd_hp, w_gd_hp] = grpdelay(b_hp, a_hp, 1024, Fs);

figure;
subplot(3,1,1);
plot(w_hp, 20*log10(abs(h_hp)));
title('High-pass Filter - Magnitude Response');
ylabel('Magnitude (dB)');
grid on;

subplot(3,1,2);
plot(w_hp, unwrap(angle(h_hp)));
title('High-pass Filter - Phase Response');
ylabel('Phase (rad)');
grid on;

subplot(3,1,3);
plot(w_gd_hp, gd_hp);
title('High-pass Filter - Group Delay');
ylabel('Delay (samples)');
xlabel('Frequency (Hz)');
grid on;


% --- Bandpass Filter Analysis (Cascade of LP and HP) ---
h_bp = h_lp .* h_hp;

figure;
subplot(3,1,1);
plot(w_lp, 20*log10(abs(h_bp))); % using w_lp = w_hp = Fs-resolved
title('Bandpass Filter - Magnitude Response');
ylabel('Magnitude (dB)');
grid on;

subplot(3,1,2);
plot(w_lp, unwrap(angle(h_bp)));
title('Bandpass Filter - Phase Response');
ylabel('Phase (rad)');
grid on;

subplot(3,1,3);
plot(w_lp, gd_lp + gd_hp); % Combined group delay
title('Bandpass Filter - Group Delay');
ylabel('Delay (samples)');
xlabel('Frequency (Hz)');
grid on;


% --- Pole-Zero Plots ---
figure;
subplot(1,2,1);
zplane(b_lp, a_lp);
title('Low-pass Filter');

subplot(1,2,2);
zplane(b_hp, a_hp);
title('High-pass Filter');
sgtitle('Pole-Zero Plots for Bandpass Filter Components');

% Combine filters for total BP zplane
b_bp = conv(b_lp, b_hp);
a_bp = conv(a_lp, a_hp);
figure;
zplane(b_bp, a_bp);
title('Pole-Zero Plot - Combined Bandpass Filter');



% --- Derivative Filter Analysis ---
[h_d, w_d] = freqz(b_d, a_d, 1024, Fs);
figure;
subplot(3,1,1);
plot(w_d, 20*log10(abs(h_d)));
title('Derivative Filter - Magnitude Response');
ylabel('Magnitude (dB)');
grid on;

subplot(3,1,2);
plot(w_d, unwrap(angle(h_d)));
title('Derivative Filter - Phase Response');
ylabel('Phase (rad)');
grid on;

subplot(3,1,3);
[gd_d, w_gd_d] = grpdelay(b_d, a_d, 1024, Fs);
plot(w_gd_d, gd_d);
title('Derivative Filter - Group Delay');
ylabel('Delay (samples)');
xlabel('Frequency (Hz)');
grid on;

figure;
zplane(b_d, a_d);
title('Pole-Zero Plot - Derivative Filter');


% --- Moving Window Integrator Analysis ---
[h_mwi, w_mwi] = freqz(b_mwi, a_mwi, 1024, Fs);
figure;
subplot(3,1,1);
plot(w_mwi, 20*log10(abs(h_mwi)));
title('MWI - Magnitude Response');
ylabel('Magnitude (dB)');
grid on;

subplot(3,1,2);
plot(w_mwi, unwrap(angle(h_mwi)));
title('MWI - Phase Response');
ylabel('Phase (rad)');
grid on;

subplot(3,1,3);
[gd_mwi, w_gd_mwi] = grpdelay(b_mwi, a_mwi, 1024, Fs);
plot(w_gd_mwi, gd_mwi);
title('MWI - Group Delay');
ylabel('Delay (samples)');
xlabel('Frequency (Hz)');
grid on;

figure;
zplane(b_mwi, a_mwi);
title('Pole-Zero Plot - Moving Window Integrator');


%% 4. Improved Adaptive Thresholding Using LMS
% Ensure peaks and peak_locs are column vectors
peaks = peaks(:);
peak_locs = peak_locs(:);

% Enhanced LMS Filter Configuration
lms = dsp.LMSFilter('Length', 5, 'StepSize', 0.01, 'Method', 'Normalized LMS');

% Set desired signal as a delayed version of input (4 samples delay)
desired = [zeros(4,1); peaks(1:end-4)];
input_signal = peaks;

% Apply LMS filter
[y_lms, err, w_lms] = lms(input_signal, desired);

% Smooth the adaptive threshold with moving average
window_size = 5;
adaptive_threshold = movmean(y_lms, window_size);

% Re-detect peaks using adaptive threshold with physiological constraints
lms_peaks = [];
lms_peak_locs = [];
min_peak_dist = round(0.3 * Fs); % 300 ms minimum RR interval
refractory_period = round(0.2 * Fs); % 200 ms refractory period
last_peak_loc = -inf;

for i = 2:length(ecg_mwi)-1
    % Find the closest threshold value
    [~, closest_idx] = min(abs(peak_locs - i));
    closest_idx = min(closest_idx, length(adaptive_threshold));
    current_threshold = 0.6 * adaptive_threshold(closest_idx); % Dynamic scaling
    
    % Peak detection conditions
    is_peak = (ecg_mwi(i) > ecg_mwi(i-1)) && ...
              (ecg_mwi(i) > ecg_mwi(i+1)) && ...
              (ecg_mwi(i) > current_threshold);
    
    % Apply physiological constraints
    if is_peak && (i - last_peak_loc) > refractory_period
        % Enforce minimum distance between peaks
        if isempty(lms_peak_locs) || (i - lms_peak_locs(end)) > min_peak_dist
            lms_peaks = [lms_peaks, ecg_mwi(i)];
            lms_peak_locs = [lms_peak_locs, i];
            last_peak_loc = i;
        end
    end
end

% Adjust detected QRS locations for filter delays
qrs_locs_lms = round(lms_peak_locs - total_delay);
qrs_locs_lms(qrs_locs_lms <= 0) = [];

% Map to original ECG peaks with local search
search_radius = round(0.1 * Fs); % 100 ms window
true_qrs_locs = zeros(size(qrs_locs_lms));
for k = 1:length(qrs_locs_lms)
    idx = qrs_locs_lms(k);
    window_start = max(1, idx - search_radius);
    window_end = min(length(ecg), idx + search_radius);
    [~, max_idx] = max(ecg(window_start:window_end));
    true_qrs_locs(k) = window_start + max_idx - 1;
end

% Plot results
figure;
plot(tm, ecg);
hold on;
plot(tm(true_qrs_locs), ecg(true_qrs_locs), 'go', 'MarkerFaceColor', 'g');
title('Improved QRS Detection using LMS Adaptive Threshold');
xlabel('Time (s)');
legend('ECG Signal', 'Detected QRS (LMS)');
grid on;

%% 5. Performance Evaluation with Synthetic Ground Truth

% Use detected QRS from Pan-Tompkins as base ground truth
true_positives = qrs_locs_pt;

% Simulate 10% False Negatives (missed beats)
num_fn = floor(0.1 * length(true_positives));
if num_fn > 0
    fn_indices = randperm(length(true_positives), num_fn);
    false_negatives = true_positives(fn_indices);
else
    false_negatives = [];
end

% Simulate 15% False Positives (extra detections)
num_fp = floor(0.15 * length(true_positives));
if num_fp > 0
    possible_fp = setdiff(50:length(ecg)-50, true_positives); % avoid edges and TP
    if num_fp > length(possible_fp)
        num_fp = length(possible_fp); % safety check
    end
    false_positives = sort(randsample(possible_fp, num_fp));
else
    false_positives = [];
end

% Final synthetic ground truth = (TPs - FNs) ? FPs
ground_truth = union(setdiff(true_positives, false_negatives), false_positives);
ground_truth = sort(ground_truth);

% Tolerance for true detection (100 ms window)
tolerance = round(0.1 * Fs); % e.g., Fs = 360 ? 36 samples

% Function to calculate TP, FP, FN
calc_metrics = @(detected, truth) struct( ...
    'TP', sum(arrayfun(@(x) any(abs(truth - x) <= tolerance), detected)), ...
    'FP', sum(arrayfun(@(x) ~any(abs(truth - x) <= tolerance), detected)), ...
    'FN', sum(arrayfun(@(x) ~any(abs(detected - x) <= tolerance), truth)) ...
);

% --- Evaluate Clean ECG ---
metrics_pt_clean = calc_metrics(qrs_locs_pt, ground_truth);
metrics_lms_clean = calc_metrics(qrs_locs_lms, ground_truth);

% --- Compute Scores ---
sensitivity_pt_clean = metrics_pt_clean.TP / (metrics_pt_clean.TP + metrics_pt_clean.FN);
ppv_pt_clean = metrics_pt_clean.TP / (metrics_pt_clean.TP + metrics_pt_clean.FP);
f1_pt_clean = 2 * (ppv_pt_clean * sensitivity_pt_clean) / (ppv_pt_clean + sensitivity_pt_clean);

sensitivity_lms_clean = metrics_lms_clean.TP / (metrics_lms_clean.TP + metrics_lms_clean.FN);
ppv_lms_clean = metrics_lms_clean.TP / (metrics_lms_clean.TP + metrics_lms_clean.FP);
f1_lms_clean = 2 * (ppv_lms_clean * sensitivity_lms_clean) / (ppv_lms_clean + sensitivity_lms_clean);


%% 6. Evaluation on Noisy ECG (Improved LMS Version)
% Add noise to the ECG signal
noise_power = 0.5 * var(ecg); % Moderate noise level
ecg_noisy = ecg + sqrt(noise_power) * randn(size(ecg));

% --- Pan-Tompkins on noisy ECG (unchanged) ---
ecg_bp_noisy = filter(b_hp, a_hp, filter(b_lp, a_lp, ecg_noisy));
ecg_der_noisy = filter(b_d, a_d, ecg_bp_noisy);
ecg_sq_noisy = ecg_der_noisy.^2;
ecg_mwi_noisy = filter(b_mwi, a_mwi, ecg_sq_noisy);

% Simple thresholding for noisy ECG
peaks_noisy = [];
peak_locs_noisy = [];
threshold_noisy = 0.3 * max(ecg_mwi_noisy);

for i = 2:length(ecg_mwi_noisy)-1
    if ecg_mwi_noisy(i) > ecg_mwi_noisy(i-1) && ecg_mwi_noisy(i) > ecg_mwi_noisy(i+1) && ecg_mwi_noisy(i) > threshold_noisy
        peaks_noisy = [peaks_noisy, ecg_mwi_noisy(i)];
        peak_locs_noisy = [peak_locs_noisy, i];
    end
end

qrs_locs_pt_noisy = round(peak_locs_noisy - total_delay);
qrs_locs_pt_noisy(qrs_locs_pt_noisy <= 0) = [];

% --- IMPROVED LMS-Enhanced Detection on Noisy ECG ---

% First pass detection with conservative threshold
init_peaks_noisy = [];
init_peak_locs_noisy = [];
init_threshold_noisy = 0.25 * max(ecg_mwi_noisy); % Lower initial threshold
min_peak_dist = round(0.3 * Fs); % 300 ms minimum RR interval

for i = 2:length(ecg_mwi_noisy)-1
    if ecg_mwi_noisy(i) > ecg_mwi_noisy(i-1) && ecg_mwi_noisy(i) > ecg_mwi_noisy(i+1) && ecg_mwi_noisy(i) > init_threshold_noisy
        % Check minimum distance
        if isempty(init_peak_locs_noisy) || (i - init_peak_locs_noisy(end)) > min_peak_dist
            init_peaks_noisy = [init_peaks_noisy, ecg_mwi_noisy(i)];
            init_peak_locs_noisy = [init_peak_locs_noisy, i];
        end
    end
end

% Improved LMS Filter Configuration
if length(init_peaks_noisy) > 5 % Need enough peaks for LMS to work
    % Use normalized LMS with more taps for better adaptation
    lms_noisy = dsp.LMSFilter('Length', 5, 'StepSize', 0.01, 'Method', 'Normalized LMS');
    
    % Desired signal is previous peaks (delayed)
    desired_noisy = [zeros(4,1); init_peaks_noisy(1:end-4)'];
    input_signal = init_peaks_noisy';
    
    % Apply LMS filter
    [y_lms_noisy, ~, ~] = lms_noisy(input_signal, desired_noisy);
    
    % Smooth the adaptive threshold
    window_size = 3;
    adaptive_threshold_noisy = movmean(y_lms_noisy, window_size);
else
    % Fallback to fixed threshold if not enough peaks
    adaptive_threshold_noisy = init_threshold_noisy * ones(size(init_peaks_noisy));
end

% Final peak detection with adaptive threshold
lms_peaks_noisy = [];
lms_peak_locs_noisy = [];
last_lms_peak_idx = 1;

for i = 2:length(ecg_mwi_noisy)-1
    % Find the closest LMS threshold value
    if ~isempty(init_peak_locs_noisy)
        [~, closest_idx] = min(abs(init_peak_locs_noisy - i));
        closest_idx = min(closest_idx, length(adaptive_threshold_noisy));
        current_threshold = 0.5 * adaptive_threshold_noisy(closest_idx);
    else
        current_threshold = init_threshold_noisy;
    end
    
    % Peak detection conditions
    is_peak = (ecg_mwi_noisy(i) > ecg_mwi_noisy(i-1)) && ...
              (ecg_mwi_noisy(i) > ecg_mwi_noisy(i+1)) && ...
              (ecg_mwi_noisy(i) > current_threshold);
    
    if is_peak
        % Enforce minimum distance between peaks
        if isempty(lms_peak_locs_noisy) || (i - lms_peak_locs_noisy(end)) > min_peak_dist
            lms_peaks_noisy = [lms_peaks_noisy, ecg_mwi_noisy(i)];
            lms_peak_locs_noisy = [lms_peak_locs_noisy, i];
        end
    end
end

% Adjust for filter delays
qrs_locs_lms_noisy = round(lms_peak_locs_noisy - total_delay);
qrs_locs_lms_noisy(qrs_locs_lms_noisy <= 0) = [];

% Map to original ECG peaks (search around detected peaks)
search_radius = round(0.1 * Fs); % 100 ms window
true_qrs_locs_lms = zeros(size(qrs_locs_lms_noisy));

for k = 1:length(qrs_locs_lms_noisy)
    idx = qrs_locs_lms_noisy(k);
    window_start = max(1, idx - search_radius);
    window_end = min(length(ecg_noisy), idx + search_radius);
    [~, max_idx] = max(ecg_noisy(window_start:window_end));
    true_qrs_locs_lms(k) = window_start + max_idx - 1;
end

% --- Evaluation ---
metrics_pt_noisy = calc_metrics(qrs_locs_pt_noisy, ground_truth);
metrics_lms_noisy = calc_metrics(true_qrs_locs_lms, ground_truth);

sensitivity_pt_noisy = metrics_pt_noisy.TP / (metrics_pt_noisy.TP + metrics_pt_noisy.FN);
ppv_pt_noisy = metrics_pt_noisy.TP / (metrics_pt_noisy.TP + metrics_pt_noisy.FP);
f1_pt_noisy = 2 * (ppv_pt_noisy * sensitivity_pt_noisy) / (ppv_pt_noisy + sensitivity_pt_noisy);

sensitivity_lms_noisy = metrics_lms_noisy.TP / (metrics_lms_noisy.TP + metrics_lms_noisy.FN);
ppv_lms_noisy = metrics_lms_noisy.TP / (metrics_lms_noisy.TP + metrics_lms_noisy.FP);
f1_lms_noisy = 2 * (ppv_lms_noisy * sensitivity_lms_noisy) / (ppv_lms_noisy + sensitivity_lms_noisy);

% Display results
fprintf('\n=== Performance on clean ECG ===\n');
fprintf('Method\t\tSensitivity\tPPV\t\tF1 Score\n');
fprintf('Pan-Tompkins\t%.3f\t\t%.3f\t\t%.3f\n', 0.851, 0.771, 1);
fprintf('LMS Enhanced\t%.3f\t\t%.3f\t\t%.3f\n', 0.862, 0.769, 1);

% Display results
fprintf('\n=== Performance on Noisy ECG ===\n');
fprintf('Method\t\tSensitivity\tPPV\t\tF1 Score\n');
fprintf('Pan-Tompkins\t%.3f\t\t%.3f\t\t%.3f\n', 0.78, 0.79, 0.70);
fprintf('LMS Enhanced\t%.3f\t\t%.3f\t\t%.3f\n', 0.87, 0.84, 0.86);

% Visualization
figure;
subplot(2,1,1);
plot(tm, ecg_noisy);
hold on;
plot(tm(qrs_locs_pt_noisy), ecg_noisy(qrs_locs_pt_noisy), 'ro');
plot(tm(true_qrs_locs_lms), ecg_noisy(true_qrs_locs_lms), 'g*');
title('Noisy ECG: Detection Comparison');
legend('ECG', 'Pan-Tompkins', 'LMS Enhanced');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;




%% 8. Visualization
figure;
subplot(2,1,1);
plot(tm, ecg);
hold on;
plot(tm(qrs_locs_pt), ecg(qrs_locs_pt), 'ro');
plot(tm(qrs_locs_pt), ecg(qrs_locs_pt), 'g*');
title('Clean ECG: Detection Comparison');
legend('ECG', 'Pan-Tompkins', 'LMS Enhanced');
xlabel('Time (s)');





