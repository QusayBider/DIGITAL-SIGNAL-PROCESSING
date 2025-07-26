
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


%% 4. Adaptive Thresholding Using LMS

%  The output of this filter will be our adaptive threshold.

% We will use the detected peaks from the MWI stage as input to the LMS.
lms = dsp.LMSFilter(1, 'StepSize', 0.05); % Length=1, mu=0.05
desired = [0; peaks(1:end-1)']; 
[y_lms, err, w_lms] = lms(peaks', desired);

adaptive_threshold = y_lms;

% Now, re-detect peaks using the adaptive threshold
lms_peaks = [];
lms_peak_locs = [];
adaptive_threshold_full = zeros(1, length(ecg_mwi));
last_lms_peak_idx = 1;

for i = 2:length(ecg_mwi)-1
    % Find the threshold corresponding to the most recent detected QRS
    if ~isempty(peak_locs) && i > peak_locs(last_lms_peak_idx) && last_lms_peak_idx < length(peaks)
        last_lms_peak_idx = last_lms_peak_idx + 1;
    end
    current_threshold_val = adaptive_threshold(last_lms_peak_idx);
    
    % Use a fraction of the predicted peak height as the threshold
    current_threshold = 0.5 * current_threshold_val;
    adaptive_threshold_full(i) = current_threshold;

    if ecg_mwi(i) > ecg_mwi(i-1) && ecg_mwi(i) > ecg_mwi(i+1) && ecg_mwi(i) > current_threshold
        lms_peaks = [lms_peaks, ecg_mwi(i)];
        lms_peak_locs = [lms_peak_locs, i];
    end
end

% Adjust locations for delay
qrs_locs_lms = round(lms_peak_locs - total_delay);
qrs_locs_lms(qrs_locs_lms <= 0) = [];

figure;
plot(tm, ecg);
hold on;
plot(tm(qrs_locs_lms), ecg(qrs_locs_lms), 'go', 'MarkerFaceColor', 'g');
title('QRS Detection using LMS Adaptive Threshold');
xlabel('Time (s)');
legend('ECG Signal', 'Detected QRS (LMS)');


%% 5. Performance Evaluation with Synthetic Ground Truth
% based on the Pan-Tompkins detections with some variations

true_positives = qrs_locs_pt;
false_negatives = true_positives(randperm(length(true_positives), floor(length(true_positives)*0.1))); % 10% FN
false_positives = randperm(length(ecg), floor(length(true_positives)*0.15)); % 15% FP
false_positives(false_positives < 50 | false_positives > length(ecg)-50) = []; % Remove edge cases

ground_truth = union(setdiff(true_positives, false_negatives), false_positives);
ground_truth = sort(ground_truth);

% Tolerance window (100ms)
tolerance = round(0.1 * Fs);

% Function to calculate metrics
calc_metrics = @(detected, truth) ...
    struct(...
    'TP', sum(arrayfun(@(x) any(abs(truth - x) <= tolerance), detected)), ...
    'FP', sum(arrayfun(@(x) ~any(abs(truth - x) <= tolerance), detected)), ...
    'FN', sum(arrayfun(@(x) ~any(abs(detected - x) <= tolerance), truth)) ...
    );

% Calculate metrics for both methods on clean ECG
metrics_pt_clean = calc_metrics(qrs_locs_pt, ground_truth);
metrics_lms_clean = calc_metrics(qrs_locs_lms, ground_truth);

% Calculate performance metrics
sensitivity_pt_clean = metrics_pt_clean.TP / (metrics_pt_clean.TP + metrics_pt_clean.FN);
ppv_pt_clean = metrics_pt_clean.TP / (metrics_pt_clean.TP + metrics_pt_clean.FP);
f1_pt_clean = 2 * (ppv_pt_clean * sensitivity_pt_clean) / (ppv_pt_clean + sensitivity_pt_clean);

sensitivity_lms_clean = metrics_lms_clean.TP / (metrics_lms_clean.TP + metrics_lms_clean.FN);
ppv_lms_clean = metrics_lms_clean.TP / (metrics_lms_clean.TP + metrics_lms_clean.FP);
f1_lms_clean = 2 * (ppv_lms_clean * sensitivity_lms_clean) / (ppv_lms_clean + sensitivity_lms_clean);

%% 6. Evaluation on Noisy ECG
% Add noise to the ECG signal
noise_power = 0.5 * var(ecg); % Moderate noise level
ecg_noisy = ecg + sqrt(noise_power) * randn(size(ecg));

% Run both algorithms on noisy ECG
% --- Pan-Tompkins on noisy ECG ---
ecg_bp_noisy = filter(b_hp, a_hp, filter(b_lp, a_lp, ecg_noisy));
ecg_der_noisy = filter(b_d, a_d, ecg_bp_noisy);
ecg_sq_noisy = ecg_der_noisy.^2;
ecg_mwi_noisy = filter(b_mwi, a_mwi, ecg_sq_noisy);

% Simple thresholding for noisy version
peaks_noisy = [];
peak_locs_noisy = [];
threshold_noisy = 0.3 * max(ecg_mwi_noisy); % Lower threshold for noisy signal

for i = 2:length(ecg_mwi_noisy)-1
    if ecg_mwi_noisy(i) > ecg_mwi_noisy(i-1) && ecg_mwi_noisy(i) > ecg_mwi_noisy(i+1) && ecg_mwi_noisy(i) > threshold_noisy
        peaks_noisy = [peaks_noisy, ecg_mwi_noisy(i)];
        peak_locs_noisy = [peak_locs_noisy, i];
    end
end
qrs_locs_pt_noisy = round(peak_locs_noisy - total_delay);
qrs_locs_pt_noisy(qrs_locs_pt_noisy <= 0) = [];

% --- LMS-enhanced on noisy ECG ---
% Find peaks for LMS input
peaks_for_lms_noisy = [];
peak_locs_for_lms_noisy = [];
init_threshold_noisy = 0.3 * max(ecg_mwi_noisy);

for i = 2:length(ecg_mwi_noisy)-1
    if ecg_mwi_noisy(i) > ecg_mwi_noisy(i-1) && ecg_mwi_noisy(i) > ecg_mwi_noisy(i+1) && ecg_mwi_noisy(i) > init_threshold_noisy
        peaks_for_lms_noisy = [peaks_for_lms_noisy, ecg_mwi_noisy(i)];
        peak_locs_for_lms_noisy = [peak_locs_for_lms_noisy, i];
    end
end

% Run LMS on noisy peaks
if ~isempty(peaks_for_lms_noisy)
    desired_noisy = [0; peaks_for_lms_noisy(1:end-1)'];
    [y_lms_noisy, err_noisy, w_lms_noisy] = lms(peaks_for_lms_noisy', desired_noisy);
    adaptive_threshold_noisy = y_lms_noisy;
else
    adaptive_threshold_noisy = init_threshold_noisy * ones(size(ecg_mwi_noisy));
end

% Detect with adaptive threshold
lms_peaks_noisy = [];
lms_peak_locs_noisy = [];
last_lms_peak_idx_noisy = 1;

for i = 2:length(ecg_mwi_noisy)-1
    if ~isempty(peak_locs_for_lms_noisy) && i > peak_locs_for_lms_noisy(last_lms_peak_idx_noisy) && last_lms_peak_idx_noisy < length(peaks_for_lms_noisy)
        last_lms_peak_idx_noisy = last_lms_peak_idx_noisy + 1;
    end
    current_threshold_val_noisy = adaptive_threshold_noisy(last_lms_peak_idx_noisy);
    current_threshold_noisy = 0.4 * current_threshold_val_noisy; % More conservative threshold

    if ecg_mwi_noisy(i) > ecg_mwi_noisy(i-1) && ecg_mwi_noisy(i) > ecg_mwi_noisy(i+1) && ecg_mwi_noisy(i) > current_threshold_noisy
        lms_peaks_noisy = [lms_peaks_noisy, ecg_mwi_noisy(i)];
        lms_peak_locs_noisy = [lms_peak_locs_noisy, i];
    end
end

qrs_locs_lms_noisy = round(lms_peak_locs_noisy - total_delay);
qrs_locs_lms_noisy(qrs_locs_lms_noisy <= 0) = [];

% Calculate metrics for noisy ECG
metrics_pt_noisy = calc_metrics(qrs_locs_pt_noisy, ground_truth);
metrics_lms_noisy = calc_metrics(qrs_locs_lms_noisy, ground_truth);

sensitivity_pt_noisy = metrics_pt_noisy.TP / (metrics_pt_noisy.TP + metrics_pt_noisy.FN);
ppv_pt_noisy = metrics_pt_noisy.TP / (metrics_pt_noisy.TP + metrics_pt_noisy.FP);
f1_pt_noisy = 2 * (ppv_pt_noisy * sensitivity_pt_noisy) / (ppv_pt_noisy + sensitivity_pt_noisy);

sensitivity_lms_noisy = metrics_lms_noisy.TP / (metrics_lms_noisy.TP + metrics_lms_noisy.FN);
ppv_lms_noisy = metrics_lms_noisy.TP / (metrics_lms_noisy.TP + metrics_lms_noisy.FP);
f1_lms_noisy = 2 * (ppv_lms_noisy * sensitivity_lms_noisy) / (ppv_lms_noisy + sensitivity_lms_noisy);

%% 7. Display Results
fprintf('\n=== Performance on Clean ECG ===\n');
fprintf('Method\t\tSensitivity\tPPV\t\tF1 Score\n');
fprintf('Pan-Tompkins\t%.3f\t\t%.3f\t\t%.3f\n', sensitivity_pt_clean, ppv_pt_clean, f1_pt_clean);
fprintf('LMS Enhanced\t%.3f\t\t%.3f\t\t%.3f\n', sensitivity_lms_clean, ppv_lms_clean, f1_lms_clean);

fprintf('\n=== Performance on Noisy ECG ===\n');
fprintf('Method\t\tSensitivity\tPPV\t\tF1 Score\n');
fprintf('Pan-Tompkins\t%.3f\t\t%.3f\t\t%.3f\n', sensitivity_pt_noisy, ppv_pt_noisy, f1_pt_noisy);
fprintf('LMS Enhanced\t%.3f\t\t%.3f\t\t%.3f\n', sensitivity_lms_noisy, ppv_lms_noisy, f1_lms_noisy);

%% 8. Visualization
figure;
subplot(2,1,1);
plot(tm, ecg);
hold on;
plot(tm(qrs_locs_pt), ecg(qrs_locs_pt), 'ro');
plot(tm(qrs_locs_lms), ecg(qrs_locs_lms), 'g*');
plot(tm(ground_truth), ecg(ground_truth), 'kx', 'MarkerSize', 8, 'LineWidth', 1.5);
title('Clean ECG: Detection Comparison');
legend('ECG', 'Pan-Tompkins', 'LMS Enhanced', 'Ground Truth');
xlabel('Time (s)');

subplot(2,1,2);
plot(tm, ecg_noisy);
hold on;
plot(tm(qrs_locs_pt_noisy), ecg_noisy(qrs_locs_pt_noisy), 'ro');
plot(tm(qrs_locs_lms_noisy), ecg_noisy(qrs_locs_lms_noisy), 'g*');
plot(tm(ground_truth), ecg_noisy(ground_truth), 'kx', 'MarkerSize', 8, 'LineWidth', 1.5);
title('Noisy ECG: Detection Comparison');
legend('ECG', 'Pan-Tompkins', 'LMS Enhanced', 'Ground Truth');
xlabel('Time (s)');

% Bar chart comparison
metrics_clean = [sensitivity_pt_clean, ppv_pt_clean, f1_pt_clean; 
                sensitivity_lms_clean, ppv_lms_clean, f1_lms_clean];
metrics_noisy = [sensitivity_pt_noisy, ppv_pt_noisy, f1_pt_noisy; 
                sensitivity_lms_noisy, ppv_lms_noisy, f1_lms_noisy];

figure;
subplot(1,2,1);
bar(metrics_clean);
set(gca, 'XTickLabel', {'Pan-Tompkins', 'LMS Enhanced'});
legend('Sensitivity', 'PPV', 'F1 Score');
title('Performance on Clean ECG');
ylim([0 1.1]);

subplot(1,2,2);
bar(metrics_noisy);
set(gca, 'XTickLabel', {'Pan-Tompkins', 'LMS Enhanced'});
legend('Sensitivity', 'PPV', 'F1 Score');
title('Performance on Noisy ECG');
ylim([0 1.1]);

