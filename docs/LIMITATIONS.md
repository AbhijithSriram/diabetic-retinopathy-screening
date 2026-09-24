# Limitations (read before quoting any number)

1. **Public data is not field data.** Every accuracy figure comes from public
   datasets taken on desktop fundus cameras (Kowa, Topcon, the DDR and APTOS
   clinic mix). None comes from the low-cost handheld cameras the PS targets.
   mBRSET (handheld, Brazil) is the closest public proxy and needs PhysioNet
   credentialed access. We make no claim about performance on Indian field
   cameras.
2. **Reference grades are human grades.** IDRiD, DDR and APTOS labels come
   from one grader or a small panel, and label noise caps measurable
   accuracy. Messidor-2's adjudicated grades (Krause et al. 2018) are the
   cleanest reference we use.
3. **The conformal guarantees assume exchangeability.** They hold on data
   that looks like the calibration split. On a new camera they can fail;
   `results/VALIDATION.md` shows how far they drift on the external sets, and
   how many audit labels restore them. We do not claim they hold on an
   unseen camera without that per-site recalibration.
4. **Single-field photos cannot apply the full 4-2-1 rule.** Venous beading
   and IRMA are not detected, and the four quadrants are only partly visible
   in a 45° macula-centred field. The rule engine can support "severe NPDR"
   but never rule it out.
5. **Neovascularisation is not segmented.** The only public pixel labels
   (FGADR, 49 masks) are available by request only. Proliferative disease is
   left to the grader and the reviewer.
6. **Macular oedema from a 2-D photo is a proxy.** The DME rule (hard
   exudates within one disc diameter of the fovea) follows the IDRiD and
   Messidor-2 definitions. It is not an OCT diagnosis.
7. **The 30-second review time is a target, not a measurement,** until the
   reader study (`/study`) is run with real ophthalmologists. The district
   simulation uses 30 s as a parameter.
8. **District simulation parameters marked ASSUMPTION in `sim/params.yaml`**
   are placeholders. Referral completion "with engine" (40%) is a target, not
   a result. The only measured figure is the 13–17% baseline from the Mohali
   trial.
9. **Translations need review.** The Hindi and Tamil messages were written by
   the developers and must be checked by native speakers who train field
   staff before any use with patients.
10. **MATLAB and Simulink code is untested.** The build machine has no MATLAB
    licence. The Python pipeline is the reference implementation; the ONNX
    export includes a numerical parity check against it.
11. **The rule engine explains, it does not triage.** A matched-workload
    ablation showed it adds no safety over widening the CNN's own review
    band, and its lesion evidence has low specificity (a single false
    component flips "more than microaneurysms alone"). Lesion counts and
    overlays are filtered to components that are real at least ~75% of the
    time on development masks; the remaining false components are the main
    open technical item.
12. **IDRiD was scored during development** (two grader runs, plus the
    lesion-confidence diagnosis), so it is no longer a pristine test set.
    Messidor-2 has been scored once per final candidate (four models over
    19–24 September), each time after the model was committed to in writing;
    every look is in the register at the end of `EXPERIMENTS.md`. Four looks
    is not one, and a sceptical reader should discount the final figures
    slightly for it --- though no model was ever chosen by its sealed-set
    score.
13. **No real patient data was used, and none may be used** without ethics
    committee approval, informed consent and compliance with the DPDP Act
    2023.
