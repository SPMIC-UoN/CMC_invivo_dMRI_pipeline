#!/usr/bin/env bash
set -euo pipefail

# Convert human in-vivo DICOM diffusion data for CMC_invivo_dMRI_pipeline.
# Keeps individual DWI runs so that complex denoising and run-level QC happen
# before the main pipeline combines AP and PA data.
#
# S05 selection:
#   AP magnitude: MR-SE012, MR-SE028
#   AP phase:     MR-SE013, MR-SE029
#   PA magnitude: MR-SE019, MR-SE035
#   PA phase:     MR-SE020, MR-SE036
# SBRefs, FA/ColFA maps, localizers, B1 maps, and MP2RAGE images are excluded
# from the diffusion inputs. The MP2RAGE INV2 image is copied to T1_2.

usage() {
    echo "Usage: $(basename "$0") <DICOM_DIR> <OUT_DIR>"
    exit 2
}

[[ $# -eq 2 ]] || usage

dicomdir="$1"
outdir="$2"

module load fsl dcm2niix

mkdir -p "${outdir}"/{AP,PA,AP_ph,PA_ph,T1}
tmpdir="${outdir}/.tmp_dcm2niix"
mkdir -p "${tmpdir}"

shopt -s nullglob

is_derived_or_reference() {
    local name="$1"
    [[ "${name}" == *SBRef* || "${name}" == *FA* || "${name}" == *ColFA* || \
       "${name}" == *ADC* || "${name}" == *TRACE* || "${name}" == *Tensor* ]]
}

is_faulty_s05_run() {
    local name="$1"
    [[ "$(basename "${outdir}")" == pt89_diff_S05 && \
       ( "${name}" == MR-SE008-* || "${name}" == MR-SE009-* ) ]]
}

direction() {
    local name="$1"
    [[ "${name}" =~ (^|[_-])AP([_-]|$) ]] && { echo AP; return; }
    [[ "${name}" =~ (^|[_-])PA([_-]|$) ]] && { echo PA; return; }
    return 1
}

convert_run() {
    local src="$1"
    local outstem="$2"
    local name
    name="$(basename "${src}")"

    local work="${tmpdir}/${name}"
    rm -rf "${work}"
    mkdir -p "${work}"

    dcm2niix -b y -z y -o "${work}" "${src}"

    local nii=("${work}"/*.nii.gz)
    [[ ${#nii[@]} -eq 1 ]] || {
        echo "error: expected one NIfTI from ${name}, found ${#nii[@]}"
        exit 1
    }

    local stem="${nii[0]%.nii.gz}"
    cp "${nii[0]}" "${outstem}.nii.gz"
    [[ -f "${stem}.json" ]] && cp "${stem}.json" "${outstem}.json"
    [[ -f "${stem}.bval" ]] && cp "${stem}.bval" "${outstem}.bval"
    [[ -f "${stem}.bvec" ]] && cp "${stem}.bvec" "${outstem}.bvec"
}

check_dwi() {
    local stem="$1"
    local nvol nbval nbvec
    nvol="$(fslval "${stem}.nii.gz" dim4)"
    nbval="$(awk '{n += NF} END {print n}' "${stem}.bval")"
    nbvec="$(awk 'NR <= 3 {n = NF} END {print n}' "${stem}.bvec")"
    [[ "${nvol}" -eq "${nbval}" && "${nvol}" -eq "${nbvec}" ]] || {
        echo "error: NIfTI/bval/bvec mismatch for ${stem}"
        exit 1
    }
}

check_phase_match() {
    local magn="$1" phase="$2"
    [[ "$(fslval "${magn}.nii.gz" dim4)" -eq "$(fslval "${phase}.nii.gz" dim4)" ]] || {
        echo "error: magnitude/phase volume mismatch: ${magn} ${phase}"
        exit 1
    }
}

ap_mag=()
pa_mag=()
ap_ph=()
pa_ph=()

for series in "${dicomdir}"/MR-SE*diff*; do
    [[ -d "${series}" ]] || continue
    name="$(basename "${series}")"

    is_faulty_s05_run "${name}" && continue
    is_derived_or_reference "${name}" && continue
    pe="$(direction "${name}")" || continue

    if [[ "${name}" == *Pha* ]]; then
        [[ "${pe}" == AP ]] && ap_ph+=("${series}") || pa_ph+=("${series}")
    else
        [[ "${pe}" == AP ]] && ap_mag+=("${series}") || pa_mag+=("${series}")
    fi
done

printf 'AP magnitude runs: %s\n' "${#ap_mag[@]}"
printf 'AP phase runs:     %s\n' "${#ap_ph[@]}"
printf 'PA magnitude runs: %s\n' "${#pa_mag[@]}"
printf 'PA phase runs:     %s\n' "${#pa_ph[@]}"

[[ ${#ap_mag[@]} -eq ${#ap_ph[@]} ]] || { echo "error: AP magnitude/phase counts differ"; exit 1; }
[[ ${#pa_mag[@]} -eq ${#pa_ph[@]} ]] || { echo "error: PA magnitude/phase counts differ"; exit 1; }

# The pipeline discovers phase files as AP_ph_*.nii.gz / PA_ph_*.nii.gz.
for i in "${!ap_mag[@]}"; do
    n=$((i + 1))
    convert_run "${ap_mag[i]}" "${outdir}/AP/AP_${n}"
    convert_run "${ap_ph[i]}" "${outdir}/AP_ph/AP_ph_${n}"
    check_dwi "${outdir}/AP/AP_${n}"
    check_phase_match "${outdir}/AP/AP_${n}" "${outdir}/AP_ph/AP_ph_${n}"
done

for i in "${!pa_mag[@]}"; do
    n=$((i + 1))
    convert_run "${pa_mag[i]}" "${outdir}/PA/PA_${n}"
    convert_run "${pa_ph[i]}" "${outdir}/PA_ph/PA_ph_${n}"
    check_dwi "${outdir}/PA/PA_${n}"
    check_phase_match "${outdir}/PA/PA_${n}" "${outdir}/PA_ph/PA_ph_${n}"
done

# Retain MP2RAGE INV2 for later masking/registration.
t1=("${dicomdir}"/MR-SE*t1_mp2rage*INV2*)
if [[ ${#t1[@]} -eq 1 ]]; then
    convert_run "${t1[0]}" "${outdir}/T1/T1_2"
    bet2 "${outdir}/T1/T1_2.nii.gz" "${outdir}/T1/T1_2_brain" -R -m -f 0.5
else
    echo "warning: expected one MP2RAGE INV2 series, found ${#t1[@]}"
fi

# Fast visual sanity check of the AP_1 b-vector orientation.
testdir="${outdir}/test_dtifit"
mkdir -p "${testdir}"
fslmaths "${outdir}/AP/AP_1.nii.gz" -Tmean -mul 0 -add 1 -bin "${testdir}/all_voxels_mask"
dtifit -k "${outdir}/AP/AP_1.nii.gz" -o "${testdir}/dti_AP_1" -m "${testdir}/all_voxels_mask" -r "${outdir}/AP/AP_1.bvec" -b "${outdir}/AP/AP_1.bval" --sse

echo "Done: ${outdir}"
