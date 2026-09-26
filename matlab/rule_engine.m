function r = rule_engine(lesions, od, fovea, lesionSize)
%RULE_ENGINE  ICDR evidence, quadrant counts and macular-oedema risk from
%   detected lesions. Mirrors drishti/rules/icdr.py.
%   lesions: struct array with fields kind ('MA','HE','EX','SE'), x, y (0-based,
%   lesionSize frame). od: [x y radius] or []. fovea: [x y] or [] (image centre).

c = lesionSize / 2;
fsrc = 'detected';
if isempty(fovea), fovea = [c c]; fsrc = 'assumed at image centre'; end
fx = fovea(1); fy = fovea(2);
if ~isempty(od) && 2 * od(3) >= lesionSize / 12 && 2 * od(3) <= lesionSize / 4
    dd = 2 * od(3); nasal = sign(od(1) - fx); ddsrc = 'measured';
else
    dd = lesionSize / 7.5; nasal = 1; ddsrc = 'assumed (disc not segmented)';
end
kinds = {'MA', 'HE', 'EX', 'SE'};
for k = 1:4, r.counts.(kinds{k}) = sum(strcmp({lesions.kind}, kinds{k})); end
qn = {'superior_nasal', 'superior_temporal', 'inferior_nasal', 'inferior_temporal'};
for k = 1:4, r.quadrants.(qn{k}) = struct('MA', 0, 'HE', 0, 'EX', 0, 'SE', 0); end
for i = 1:numel(lesions)
    L = lesions(i);
    if L.y < fy, v = 'superior'; else, v = 'inferior'; end
    if sign(L.x - fx) == nasal, h = 'nasal'; else, h = 'temporal'; end
    key = [v '_' h];
    r.quadrants.(key).(L.kind) = r.quadrants.(key).(L.kind) + 1;
end
trace = {};
if isempty(lesions)
    r.evidence_grade = 0; trace{end + 1} = 'No lesions detected.';
elseif all(strcmp({lesions.kind}, 'MA'))
    r.evidence_grade = 1; trace{end + 1} = sprintf('Only microaneurysms (%d): consistent with mild NPDR.', r.counts.MA);
else
    r.evidence_grade = 2; trace{end + 1} = 'More than microaneurysms alone: at least moderate NPDR (referable).';
end
heAll = all(cellfun(@(k) r.quadrants.(k).HE > 20, qn));
if heAll
    r.evidence_grade = 3; trace{end + 1} = '>20 haemorrhages in all four quadrants: 4-2-1 haemorrhage criterion met.';
end
trace{end + 1} = 'Not assessed from this photo: venous beading, IRMA, neovascularisation.';
ex = lesions(strcmp({lesions.kind}, 'EX'));
if ~isempty(ex)
    d = min(hypot([ex.x] - fx, [ex.y] - fy)) / dd;
    r.dme_risk = 1 + (d <= 1);
    r.nearest_exudate_dd = round(d, 2);
    trace{end + 1} = sprintf('Nearest hard exudate %.1f disc diameters from the fovea.', d);
else
    r.dme_risk = 0;
    r.nearest_exudate_dd = NaN;
end
r.lesion_flag = r.evidence_grade >= 2;
r.dme_flag = r.dme_risk == 2;
r.trace = trace;
r.fovea = struct('x', fx, 'y', fy, 'source', fsrc);
r.disc_diameter_px = dd;
r.disc_diameter_source = ddsrc;
end
