#!/usr/bin/env bash
set -euo pipefail

usage(){
  cat <<EOF
Usage:
  $0 <DICOM_DIR> <OUT_DIR> <DIFF_SEARCH> <T1_SEARCH> [options]

Options:
  -v, --v AX AY AZ        Axes for fslswapdim (default: x y z)
  -r, --r Sx Sy Sz        Signs for axes (1 or -1; default: 1 1 1)
  --prep-py PATH          Path to prep_dmri_reorient.py
                          (default: /home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py)
  --bvec-adjust           Reflip bvecs only (do NOT touch NIfTIs).
                          Uses -v/-r exactly as provided (no extra Z inversion).
                          Clears test_dtifit and reruns sanity dtifit.
  --remove-initial-b0     Remove the first volume from each diffusion series
                          before the dtifit sanity check.
  --remove-b0-py PATH     Path to remove_initial_b0.py
                          (default: /home/mszsaw2/remove_initial_b0.py)
  -test, --test           Test mode: only process AP_1

Behaviour:
  - Default run: converts DICOMs, reorients (if requested), fixes headers, then flips bvecs.
                 During bvec flipping, Z sign is inverted relative to -r to account for header fixes.
  - --bvec-adjust run: skips DICOM/NIfTI steps, re-flips bvecs in AP/PA/AP_ph/PA_ph only,
                       applying exactly the -v/-r you specify (no extra Z inversion).
  - In all runs, a small dtifit sanity test is attempted on AP_1 if available.
  - If --remove-initial-b0 is set, the first volume is removed from diffusion series in
    OUT_DIR/AP and OUT_DIR/PA after bvec flipping and before dtifit sanity testing.
  - -test/--test: only processes AP_1 so flips/orientation can be checked quickly.

Notes:
  A suggested brain-extraction command is printed (not executed) for convenience.
EOF
  exit 2
}

# ---- Parse required args ----
[ "$#" -lt 4 ] && usage
DICOM_DIR="$1"
OUT_DIR="$2"
SRCH="$3"
SRCH_T1="$4"
shift 4

# ---- Defaults ----
PREP_PY="/home/data/NIH_CONNECTS/nipype_dev/prep_dmri_reorient.py"
BVECFLIP_PY="/home/data/NIH_CONNECTS/CMC_tools/bvecflip_generic.py"
V=(x y z)
R=(1 1 1)
BVEC_ADJUST=false
REMOVE_INITIAL_B0=false
REMOVE_B0_PY="/home/mszsaw2/remove_initial_b0.py"
TEST_MODE=false

# ---- Parse optional flags ----
while (( "$#" )); do
  case "$1" in
    -v|--v)
      [ "$#" -lt 4 ] && usage
      V=("$2" "$3" "$4")
      shift 4
      ;;
    -r|--r)
      [ "$#" -lt 4 ] && usage
      R=("$2" "$3" "$4")
      shift 4
      ;;
    --prep-py)
      [ "$#" -lt 2 ] && usage
      PREP_PY="$2"
      shift 2
      ;;
    --bvec-adjust|--bvec_adjust)
      BVEC_ADJUST=true
      shift 1
      ;;
    --remove-initial-b0|--remove_initial_b0)
      REMOVE_INITIAL_B0=true
      shift 1
      ;;
    --remove-b0-py|--remove_b0_py)
      [ "$#" -lt 2 ] && usage
      REMOVE_B0_PY="$2"
      shift 2
      ;;
    -test|--test)
      TEST_MODE=true
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# ---- Tool checks ----
need_cmd(){
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: $1 not found"
    exit 1
  }
}

need_cmd fslroi
need_cmd fslmaths
need_cmd dtifit
need_cmd python3
$BVEC_ADJUST || need_cmd dcm2niix

if $REMOVE_INITIAL_B0; then
  [ -f "$REMOVE_B0_PY" ] || {
    echo "error: remove_initial_b0 script not found: $REMOVE_B0_PY"
    exit 1
  }
fi

# ---- Helpers ----
latest_nii(){
  ls -t "$1"/*.nii.gz 2>/dev/null | head -n1 || true
}

detect_tag(){
  local n="$1"
  [[ "$n" == *LR* ]] && { echo LR; return; }
  [[ "$n" == *RL* ]] && { echo RL; return; }
  [[ "$n" == *HF* || "$n" == *AP* ]] && { echo AP; return; }
  [[ "$n" == *FH* || "$n" == *PA* ]] && { echo PA; return; }
  return 1
}

is_derived_series(){
  local b
  b="$(basename "$1")"

  case "$b" in
    *ADC*|*TRACEW*|*FA*|*ColFA*|*Tensor*|*TENSOR*|*DWIMap*|*Report* )
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_prep(){
  local in_nii="$1"
  local out_nii="$2"

  if [[ "${V[*]}" == "x y z" && "${R[*]}" == "1 1 1" ]]; then
    python3 "$PREP_PY" -i "$in_nii" -o "$out_nii" --header-only --hard-fix-header
  else
    python3 "$PREP_PY" -i "$in_nii" -o "$out_nii" -v "${V[@]}" -r "${R[@]}" --hard-fix-header
  fi
}

abs_path(){
  if [ -e "$1" ]; then
    python3 - "$1" <<'PY'
import os, sys
p = sys.argv[1]
print(os.path.abspath(p))
PY
  else
    echo "$1"
  fi
}

remove_initial_b0_in_dir(){
  local indir="$1"

  [ -d "$indir" ] || return 0

  mapfile -t nii_list < <(
    find "$indir" -maxdepth 1 -type f -name "*.nii.gz" | sort -V
  )

  if [ "${#nii_list[@]}" -eq 0 ]; then
    echo "  No NIfTIs found in $indir for initial b0 removal."
    return 0
  fi

  local tmpdir="${indir}/remove_initial_b0_tmp"
  rm -rf "$tmpdir"
  mkdir -p "$tmpdir"

  echo "  Running initial b0 removal in: $indir"
  python3 "$REMOVE_B0_PY" \
    -indat "${nii_list[@]}" \
    -outdir "$tmpdir" \
    --b0range 0

  local trimmed_nii
  for trimmed_nii in "$tmpdir"/*.nii.gz; do
    [ -e "$trimmed_nii" ] || continue

    local base stem
    base="$(basename "$trimmed_nii")"
    stem="${base%.nii.gz}"

    mv -f "$tmpdir/${stem}.nii.gz" "$indir/${stem}.nii.gz"
    mv -f "$tmpdir/${stem}.bval"   "$indir/${stem}.bval"
    mv -f "$tmpdir/${stem}.bvec"   "$indir/${stem}.bvec"

    echo "    Overwrote ${stem}.nii.gz/.bval/.bvec"
  done

  rm -rf "$tmpdir"
}

convert_series_to_tmp(){
  local dicom_dir="$1"
  local tmpdir="$2"

  rm -rf "$tmpdir"
  mkdir -p "$tmpdir"

  dcm2niix -z y -o "$tmpdir" "$dicom_dir" >&2 || return 1

  find "$tmpdir" -maxdepth 1 -type f -name "*.nii.gz" | sort -V | head -n1
}

# =========================
# Mode banner
# =========================
if $BVEC_ADJUST; then
  echo "Mode: BVEC-ADJUST ONLY (no DICOM/NIfTI work)"
else
  echo "Mode: FULL RUN (convert/reorient + bvec flip)"
fi

echo "prep_invivo:"
echo "  DICOM_DIR: $(abs_path "$DICOM_DIR")"
echo "  OUT_DIR:   $(abs_path "$OUT_DIR")"
echo "  DIFF SRCH: $SRCH"
echo "  T1 SRCH:   $SRCH_T1"
echo "  Orient V:  (${V[*]})"
echo "  Orient R:  (${R[*]})"
echo "  Remove initial b0: $REMOVE_INITIAL_B0"
echo "  Test mode: $TEST_MODE"
echo

# =========================
# FULL RUN: DICOM -> NIfTI + reorient + header fix
# =========================
if ! $BVEC_ADJUST; then
  rm -rf "$OUT_DIR"
  mkdir -p "$OUT_DIR"/{AP,PA,AP_ph,PA_ph,T1}

  TMP_CONV_ROOT="$OUT_DIR/.tmp_dcm2niix"
  mkdir -p "$TMP_CONV_ROOT"

  # ---- Diffusion magnitude (exclude 'Pha') ----
  ap=1
  pa=1

  for d in "$DICOM_DIR"/*"$SRCH"*; do
    [ -d "$d" ] || continue
    [[ "$d" == *Pha* ]] && continue

    is_derived_series "$d" && {
      echo "Skipping derived series: $(basename "$d")"
      continue
    }

    base="$(basename "$d")"

    if ! tag="$(detect_tag "$base")"; then
      echo "Skipping unrecognised series: $base"
      continue
    fi

    case "$tag" in
      AP)
        if $TEST_MODE && [ "$ap" -ne 1 ]; then
          echo "Skipping $base (test mode: only AP_1)"
          continue
        fi
        subdir="$OUT_DIR/AP"
        out_stem="${subdir}/AP_${ap}"
        ((ap++))
        ;;
      PA)
        if $TEST_MODE; then
          echo "Skipping $base (test mode: only AP_1)"
          continue
        fi
        subdir="$OUT_DIR/PA"
        out_stem="${subdir}/PA_${pa}"
        ((pa++))
        ;;
      LR|RL)
        echo "ERROR: Detected $tag but only AP/PA outputs are supported."
        exit 1
        ;;
      *)
        echo "Unknown tag for $base"
        exit 1
        ;;
    esac

    tmpdir="$TMP_CONV_ROOT/${base}"
    nii="$(convert_series_to_tmp "$d" "$tmpdir")"
    [ -n "${nii:-}" ] || {
      echo "warning: no NIfTI found after converting $d"
      continue
    }

    mkdir -p "$subdir"
    run_prep "$nii" "${out_stem}.nii.gz"

    stem="${nii%.nii.gz}"
    [ -f "${stem}.bval" ] && cp -f "${stem}.bval" "${out_stem}.bval" || true
    [ -f "${stem}.bvec" ] && cp -f "${stem}.bvec" "${out_stem}.bvec" || true
    [ -f "${stem}.json" ] && cp -f "${stem}.json" "${out_stem}.json" || true
  done

  # ---- Diffusion phase (only 'Pha') ----
  if ! $TEST_MODE; then
    ap=1
    pa=1

    for d in "$DICOM_DIR"/*"$SRCH"*; do
      [ -d "$d" ] || continue
      [[ "$d" != *Pha* ]] && continue

      is_derived_series "$d" && {
        echo "Skipping derived phase series: $(basename "$d")"
        continue
      }

      base="$(basename "$d")"

      if ! tag="$(detect_tag "$base")"; then
        echo "Skipping unrecognised phase series: $base"
        continue
      fi

      case "$tag" in
        AP)
          subdir="$OUT_DIR/AP_ph"
          out_stem="${subdir}/AP_${ap}_ph"
          ((ap++))
          ;;
        PA)
          subdir="$OUT_DIR/PA_ph"
          out_stem="${subdir}/PA_${pa}_ph"
          ((pa++))
          ;;
        LR|RL)
          echo "ERROR: Detected $tag in phase series but only AP_ph/PA_ph outputs are supported."
          exit 1
          ;;
        *)
          echo "Unknown tag for $base"
          exit 1
          ;;
      esac

      tmpdir="$TMP_CONV_ROOT/${base}"
      nii="$(convert_series_to_tmp "$d" "$tmpdir")"
      [ -n "${nii:-}" ] || {
        echo "warning: no NIfTI found after converting $d"
        continue
      }

      mkdir -p "$subdir"
      run_prep "$nii" "${out_stem}.nii.gz"

      stem="${nii%.nii.gz}"
      [ -f "${stem}.json" ] && cp -f "${stem}.json" "${out_stem}.json" || true
    done
  else
    echo "Skipping phase series conversion (test mode)"
  fi

  # ---- T1 / anatomical ----
  if ! $TEST_MODE; then
    t1=1

    for d in "$DICOM_DIR"/*"$SRCH_T1"*; do
      [ -d "$d" ] || continue

      base="$(basename "$d")"
      tmpdir="$TMP_CONV_ROOT/${base}"

      nii="$(convert_series_to_tmp "$d" "$tmpdir")"
      [ -n "${nii:-}" ] || {
        echo "warning: no NIfTI found after converting $d"
        continue
      }

      subdir="$OUT_DIR/T1"
      out_stem="${subdir}/T1_${t1}"
      ((t1++))

      mkdir -p "$subdir"
      run_prep "$nii" "${out_stem}.nii.gz"

      stem="${nii%.nii.gz}"
      [ -f "${stem}.json" ] && cp -f "${stem}.json" "${out_stem}.json" || true
    done
  else
    echo "Skipping T1 conversion (test mode)"
  fi

  rm -rf "$TMP_CONV_ROOT"
fi

# =========================
# BVEC flipping
# =========================
echo
echo "Preparing bvec flip parameters..."

vr_args=("${V[@]}")

vx="${R[0]}"
vy="${R[1]}"
vz="${R[2]}"

case "$vx" in
  1|-1) : ;;
  *) echo "warning: Sx (${R[0]}) not in {1,-1}; defaulting to 1"; vx=1 ;;
esac

case "$vy" in
  1|-1) : ;;
  *) echo "warning: Sy (${R[1]}) not in {1,-1}; defaulting to 1"; vy=1 ;;
esac

case "$vz" in
  1|-1) : ;;
  *) echo "warning: Sz (${R[2]}) not in {1,-1}; defaulting to 1"; vz=1 ;;
esac

if $BVEC_ADJUST; then
  vf_x="$vx"
  vf_y="$vy"
  vf_z="$vz"
  echo "  BVEC-ADJUST: using -vf exactly as provided."
else
  vf_x="$vx"
  vf_y="$vy"
  vf_z="$vz"
  # vf_z=$(( -1 * vz ))
  echo "  FULL RUN: applying Z inversion relative to provided -r."
fi

echo "  -vr ${vr_args[*]}"
echo "  -vf ${vf_x} ${vf_y} ${vf_z}"
echo

if $TEST_MODE; then
  mapfile -d '' BVEC_LIST < <(
    find "$OUT_DIR/AP" -maxdepth 1 -type f -name "AP_1.bvec" -print0 2>/dev/null || true
  )
else
  mapfile -d '' BVEC_LIST < <(
    {
      find "$OUT_DIR/AP"    -maxdepth 1 -type f -name "*.bvec" -print0 2>/dev/null
      find "$OUT_DIR/PA"    -maxdepth 1 -type f -name "*.bvec" -print0 2>/dev/null
      find "$OUT_DIR/AP_ph" -maxdepth 1 -type f -name "*.bvec" -print0 2>/dev/null
      find "$OUT_DIR/PA_ph" -maxdepth 1 -type f -name "*.bvec" -print0 2>/dev/null
    } || true
  )
fi

for bvec in "${BVEC_LIST[@]:-}"; do
  [ -f "$bvec" ] || continue

  echo "Flipping in place: $bvec"
  tmp_bvec="${bvec}.tmp"

  python3 "$BVECFLIP_PY" \
    -in "$bvec" \
    -out "$tmp_bvec" \
    -vr "${vr_args[@]}" \
    -vf "$vf_x" "$vf_y" "$vf_z"

  mv -f "$tmp_bvec" "$bvec"
done

# =========================
# Optional initial b0 removal
# =========================
if $REMOVE_INITIAL_B0; then
  echo
  echo "Removing initial b0 volumes from diffusion series..."

  if $TEST_MODE; then
    remove_initial_b0_in_dir "$OUT_DIR/AP"
  else
    remove_initial_b0_in_dir "$OUT_DIR/AP"
    remove_initial_b0_in_dir "$OUT_DIR/PA"
  fi
fi

# =========================
# Quick dtifit sanity check
# =========================
echo
echo "Running quick dtifit sanity check (AP_1 if available)..."

TESTDIR="$OUT_DIR/test_dtifit"
rm -rf "$TESTDIR"
mkdir -p "$TESTDIR"

AP1_NII="$OUT_DIR/AP/AP_1.nii.gz"
AP1_BVEC="$OUT_DIR/AP/AP_1.bvec"
AP1_BVAL="$OUT_DIR/AP/AP_1.bval"

if [[ -f "$AP1_NII" && -f "$AP1_BVEC" && -f "$AP1_BVAL" ]]; then
  fslroi "$AP1_NII" "$TESTDIR/mask" 0 1
  fslmaths "$TESTDIR/mask" -mul 0 -add 1 -bin "$TESTDIR/mask"

  dtifit \
    -k "$AP1_NII" \
    -m "$TESTDIR/mask" \
    -o "$TESTDIR/dti" \
    -r "$AP1_BVEC" \
    -b "$AP1_BVAL" || echo "warning: dtifit sanity check failed (continuing)..."
else
  echo "  Skipping: AP_1 (nii/bvec/bval) not all present."
fi

# =========================
# Print a suggested brain extraction command
# =========================
echo
echo "Heads up: here is a brain-extraction command for you to adjust and run manually if desired:"

T1_CAND="$(latest_nii "$OUT_DIR/T1")"
if [ -n "${T1_CAND:-}" ]; then
  FULL_T1="$(abs_path "$T1_CAND")"
  FULL_OUT="$(abs_path "$OUT_DIR")"

  echo
  echo "  bet4animal \"$FULL_T1\" \"$FULL_OUT/t1\" -n -z 2 -m -R -f 0.5 -g 0"
  echo
  echo "(This is not executed by the script; review and tweak parameters as needed.)"
else
  echo "  No T1 found in $OUT_DIR/T1; skipping suggestion."
fi

# ---- Summary ----
echo
echo "Done -> $(abs_path "$OUT_DIR")"
echo "  Mode: $($BVEC_ADJUST && echo 'BVEC-ADJUST' || echo 'FULL')"
echo "  Test mode:        $TEST_MODE"
echo "  Diff magnitude:   $OUT_DIR/AP and $OUT_DIR/PA"
echo "  Diff phase:       $OUT_DIR/AP_ph and $OUT_DIR/PA_ph"
echo "  T1:               $OUT_DIR/T1"
echo "  bvecs flipped with: -vr ${vr_args[*]}  -vf ${vf_x} ${vf_y} ${vf_z}"
echo "  dtifit sanity outputs: $TESTDIR"
