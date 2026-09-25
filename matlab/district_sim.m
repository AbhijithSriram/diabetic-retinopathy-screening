function S = district_sim(paramsFile, reviewers, onlineFraction, completion)
%DISTRICT_SIM  Day-stepped capacity model of a district DR programme in base
%   MATLAB (no toolboxes). Cross-check for sim/district.py and the SimEvents
%   model: same parameters, coarser time step (one day).
%
%   S = district_sim('../sim/params.json', 2, 0.6, 0.40)
%
%   Each day: arrivals (Poisson), triage split, uploads limited by link
%   hours x bandwidth, tele-reviews limited by reviewer hours / 30 s per
%   case, hospital slots. Backlogs carry over in FIFO cohorts, so the share
%   of reviews finished within the SLA can be counted.

% isempty as well as nargin, so district_sim([], 2, 0.6, 0.40) works -- that
% is the documented way to override the later arguments while keeping the
% default parameter file.
if nargin < 1 || isempty(paramsFile)
    paramsFile = fullfile(fileparts(mfilename('fullpath')), '..', 'sim', 'params.json');
end
P = jsondecode(fileread(paramsFile));
if nargin < 2 || isempty(reviewers), reviewers = P.review.reviewers; end
if nargin < 3 || isempty(onlineFraction), onlineFraction = P.network.online_fraction; end
if nargin < 4 || isempty(completion), completion = P.referral.completion_baseline; end
rng(P.run.seed);
days = P.run.days;
perDay = P.district.annual_screenings / P.district.working_days_per_year;
tri = P.triage;
uploadPerDay = P.district.sites * P.district.site_hours_per_day * onlineFraction * 3600 * P.network.bandwidth_kbps / (8 * 1024 * P.network.image_mb * P.network.eyes_per_patient);
reviewPerDay = reviewers * P.review.review_hours_per_reviewer_day * 3600 / P.review.seconds_per_case;
slaDays = P.review.sla_hours / 24;

uploadQ = zeros(0, 2);  % [arrival day, count]
reviewQ = zeros(0, 2);
hospQ = 0;
done = 0; late = 0; attended = 0; referrals = 0; reviewsTotal = 0;
for d = 1:days
    n = poissrnd_local(perDay);
    ungr = binornd_local(n, P.capture.intrinsic_ungradable);
    g = n - ungr;
    nRev = binornd_local(g, tri.review + tri.audit_fraction_of_recalls * tri.recall);
    nRef = binornd_local(g - nRev, tri.refer / max(1e-9, 1 - tri.review));
    uploadQ(end + 1, :) = [d, nRev]; %#ok<AGROW>
    [uploadQ, moved] = drain(uploadQ, uploadPerDay);
    for k = 1:size(moved, 1), reviewQ(end + 1, :) = moved(k, :); end %#ok<AGROW>
    [reviewQ, reviewed] = drain(reviewQ, reviewPerDay);
    for k = 1:size(reviewed, 1)
        wait = d - reviewed(k, 1);
        done = done + reviewed(k, 2);
        late = late + reviewed(k, 2) * (wait > slaDays);
    end
    nRevRef = binornd_local(sum(reviewed(:, 2)), P.review.refer_after_review);
    newRef = ungr + nRef + nRevRef;
    referrals = referrals + newRef;
    att = binornd_local(newRef, completion);
    attended = attended + att;
    hospQ = max(0, hospQ + att - P.referral.hospital_slots_per_day);
    reviewsTotal = reviewsTotal + nRev;
end
yr = P.district.working_days_per_year / days;
S = struct('reviewers', reviewers, 'online_fraction', onlineFraction, 'completion', completion, ...
    'reviews_per_year', round(reviewsTotal * yr), 'review_within_sla_pct', 100 * (done - late) / max(1, done + sum(reviewQ(:, 2))), ...
    'review_backlog_end', sum(reviewQ(:, 2)) + sum(uploadQ(:, 2)), 'referrals_per_year', round(referrals * yr), ...
    'attended_per_year', round(attended * yr), 'hospital_backlog_end', hospQ);
disp(S);
end

function [Q, out] = drain(Q, cap)
out = zeros(0, 2);
while cap > 0 && ~isempty(Q)
    take = min(cap, Q(1, 2));
    out(end + 1, :) = [Q(1, 1), take]; %#ok<AGROW>
    Q(1, 2) = Q(1, 2) - take;
    cap = cap - take;
    if Q(1, 2) <= 0, Q(1, :) = []; end
end
end

function k = poissrnd_local(lam)
% Poisson draw without the Statistics Toolbox (normal approximation for large lambda)
if lam > 50, k = max(0, round(lam + sqrt(lam) * randn())); return; end
L = exp(-lam); k = 0; p = 1;
while true, p = p * rand(); if p <= L, break; end, k = k + 1; end
end

function k = binornd_local(n, p)
k = sum(rand(1, max(0, round(n))) < p);
end
