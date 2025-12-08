#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Merge diffusion runs by PE direction and "ph" flag
# Produces one merged file set per class:
#   OUTDIR/AP/AP_1.{nii.gz,bval,bvec}
#   OUTDIR/AP_ph/AP_1_ph.{nii.gz,bval,bvec}
#   OUTDIR/PA/PA_1.{nii.gz,bval,bvec}
#   OUTDIR/PA_ph/PA_1_ph.{nii.gz,bval,bvec}
# Each directory also includes *_inputs.txt listing merged runs.
# Includes per-run and post-merge sanity checks.
# Requires: FSL (fslmerge, fslval), coreutils, awk, paste, sort -V
# ============================================================

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <INPUT_RAW_DIR> <OUTDIR>

Example:
  $(basename "$0") ./raw ./merged
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

INDIR="$1"
OUTDIR="$2"
mkdir -p "$OUTDIR"

# Check dependencies
for dep in fslmerge fslval awk paste sort; do
  command -v "$dep" >/dev/null 2>&1 || { echo "error: $dep not found in PATH"; exit 2; }
done

shopt -s nullglob

# Collect candidate NIfTI files
mapfile -t NII_LIST < <(find "$INDIR" -maxdepth 1 -type f -name '*.nii.gz' | sort -V)
if [[ ${#NII_LIST[@]} -eq 0 ]]; then
  echo "error: no .nii.gz files found in $INDIR"
  exit 2
fi

# Group containers
AP_files=(); AP_ph_files=(); PA_files=(); PA_ph_files=()

# ---------- helpers ----------

classify() {
  local fn="$1"
  local base="$(basename "$fn")"
  local dir_tag=""
  local ph_tag=""

  if [[ "$base" =~ (^|[^A-Za-z])AP([^A-Za-z]|$) ]]; then
    dir_tag="AP"
  elif [[ "$base" =~ (^|[^A-Za-z])PA([^A-Za-z]|$) ]]; then
    dir_tag="PA"
  else
    echo "skip"; return
  fi

  if [[ "$base" =~ (_ph|Pha|PHA|_PH) ]]; then
    ph_tag="_ph"
  else
    ph_tag=""
  fi

  echo "${dir_tag}${ph_tag}"
}

get_dim4() { fslval "$1" dim4; }

count_bvals_entries() { awk '{n+=NF} END{print n+0}' "$1"; }

count_bvecs_cols() {
  awk '
    NR==1{n1=NF}
    NR==2{n2=NF}
    NR==3{n3=NF}
    END{
      if (NR<3){print -1; exit}
      if (n1!=n2 || n1!=n3){print -2; exit}
      print n1+0
    }' "$1"
}

precheck_run() {
  local nii="$1" bval="$2" bvec="$3"

  [[ -f "$bval" && -f "$bvec" ]] || {
    echo "error: missing bval/bvec for $(basename "$nii")"
    exit 2
  }

  local d4 nb nv
  d4=$(get_dim4 "$nii")
  nb=$(count_bvals_entries "$bval")
  nv=$(count_bvecs_cols "$bvec")

  if [[ "$nv" -lt 0 ]]; then
    echo "error: malformed bvec file: $bvec"
    exit 2
  fi

  if [[ "$d4" -ne "$nb" || "$d4" -ne "$nv" ]]; then
    echo "error: volume mismatch for $(basename "$nii")"
    echo "  NIfTI dim4=$d4, bval=$nb, bvec=$nv"
    exit 2
  fi
}

merge_niftis() {
  local out="$1"; shift
  local -a inputs=("$@")
  mapfile -t inputs < <(printf '%s\n' "${inputs[@]}" | sort -V)
  [[ ${#inputs[@]} -eq 1 ]] && cp -f "${inputs[0]}" "$out" || fslmerge -t "$out" "${inputs[@]}"
}

merge_bvals() {
  local out="$1"; shift
  awk '{for(i=1;i<=NF;i++) printf "%s ", $i} END{print ""}' "$@" | awk '{$1=$1;print}' > "$out"
}

merge_bvecs() {
  local out="$1"; shift
  paste -d ' ' "$@" | awk '{$1=$1;print}' > "$out"
}

postcheck_group() {
  local label="$1" nii="$2" bval="$3" bvec="$4"
  local d4 nb nv
  d4=$(get_dim4 "$nii")
  nb=$(count_bvals_entries "$bval")
  nv=$(count_bvecs_cols "$bvec")
  if [[ "$nv" -lt 0 ]]; then echo "error: malformed merged bvec ($bvec)"; exit 2; fi
  if [[ "$d4" -ne "$nb" || "$d4" -ne "$nv" ]]; then
    echo "error: ${label} mismatch after merge (dim4=$d4, bval=$nb, bvec=$nv)"
    exit 2
  fi
  echo "OK: ${label} volumes match (dim4=$d4)"
}

report_group() {
  local name="$1"; shift
  local -a files=("$@")
  echo "Found ${#files[@]} runs for ${name}"
  for f in "${files[@]}"; do echo "  - $f"; done
}

do_group() {
  local label="$1" subdir="$2" prefix="$3"
  shift 3
  local -a nii_files=("$@")
  [[ ${#nii_files[@]} -eq 0 ]] && return

  mkdir -p "${OUTDIR}/${subdir}"
  local out_nii="${OUTDIR}/${subdir}/${prefix}.nii.gz"
  local out_bval="${OUTDIR}/${subdir}/${prefix}.bval"
  local out_bvec="${OUTDIR}/${subdir}/${prefix}.bvec"

  # Build sidecar arrays + sanity checks
  local bval_files=() bvec_files=()
  for nii in "${nii_files[@]}"; do
    local stem="${nii%.nii.gz}"
    local bval="${stem}.bval"
    local bvec="${stem}.bvec"
    precheck_run "$nii" "$bval" "$bvec"
    bval_files+=("$bval")
    bvec_files+=("$bvec")
  done

  echo "Merging ${label} -> ${out_nii}"
  merge_niftis "$out_nii" "${nii_files[@]}"
  merge_bvals  "$out_bval" "${bval_files[@]}"
  merge_bvecs  "$out_bvec" "${bvec_files[@]}"
  postcheck_group "$label" "$out_nii" "$out_bval" "$out_bvec"

  {
    echo "# Inputs merged into ${prefix}"
    printf '%s\n' "${nii_files[@]}"
  } > "${OUTDIR}/${subdir}/${prefix}_inputs.txt"
}

# ---------- classify files ----------
for nii in "${NII_LIST[@]}"; do
  key=$(classify "$nii")
  [[ "$key" == "skip" ]] && { echo "warn: skipping $(basename "$nii")"; continue; }
  case "$key" in
    AP)     AP_files+=("$nii") ;;
    AP_ph)  AP_ph_files+=("$nii") ;;
    PA)     PA_files+=("$nii") ;;
    PA_ph)  PA_ph_files+=("$nii") ;;
  esac
done

# ---------- report ----------
report_group "AP"     "${AP_files[@]}"
report_group "AP_ph"  "${AP_ph_files[@]}"
report_group "PA"     "${PA_files[@]}"
report_group "PA_ph"  "${PA_ph_files[@]}"

# ---------- merge ----------
do_group "AP"    "AP"    "AP_1"     "${AP_files[@]}"
do_group "AP_ph" "AP_ph" "AP_1_ph"  "${AP_ph_files[@]}"
do_group "PA"    "PA"    "PA_1"     "${PA_files[@]}"
do_group "PA_ph" "PA_ph" "PA_1_ph"  "${PA_ph_files[@]}"

echo "Done. Outputs written under: $OUTDIR"
