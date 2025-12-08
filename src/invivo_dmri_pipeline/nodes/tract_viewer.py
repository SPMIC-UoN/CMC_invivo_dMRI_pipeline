#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
from pathlib import Path


FSLDIR = os.getenv("FSLDIR")
if not FSLDIR:
    sys.stderr.write("error: FSLDIR not set in environment.\n")
    sys.exit(2)

FSLBIN = Path(FSLDIR) / "bin"


def run(cmd: str, verbose: bool = False) -> subprocess.CompletedProcess:
    if verbose:
        print(f"\n[fsleyes command]\n{cmd}\n")
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if verbose and proc.stderr:
        print(f"[fsleyes stderr]\n{proc.stderr.strip()}\n")
    return proc


def require_tool(name: str):
    tool = FSLBIN / name
    if not tool.exists():
        sys.stderr.write(f"error: required tool not found: {tool}\n")
        sys.exit(2)


def imtest(fname: str) -> bool:
    r = run(f'"{FSLBIN / "imtest"}" "{fname}"')
    return r.stdout.strip() == "1"


def get_cog_mm(img_path: str) -> tuple[float, float, float]:
    """
    Use fslstats to get centre-of-gravity (COG) in mm coordinates
    for the provided image (typically a brain/brainmask).
    """
    cmd = [
        str(FSLBIN / "fslstats"),
        img_path,
        "-c",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
    vals = proc.stdout.strip().split()
    if len(vals) < 3:
        raise RuntimeError(f"Unexpected fslstats -c output: {proc.stdout}")
    return float(vals[0]), float(vals[1]), float(vals[2])


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"error: {message}\n\n")
        self.print_help()
        sys.exit(2)


def parse_args():
    p = MyParser(prog="view xtract results")
    req = p.add_argument_group("Required arguments")

    req.add_argument("--xtract", metavar="<folder>", required=True,
                     help="Path to XTRACT output folder (contains tract subfolders)")
    req.add_argument("--brain", metavar="<nifti>", required=True,
                     help="Template brain/brainmask to display underneath the tracts (e.g., NMT)")

    p.add_argument("--outfile", metavar="<png-or-dir>", default=None,
                   help="If set, render PNGs to this file/dir via `fsleyes render`; "
                        "otherwise open interactive `fsleyes`.")

    p.add_argument("--xzoom", type=int, default=85)
    p.add_argument("--yzoom", type=int, default=85)
    p.add_argument("--zzoom", type=int, default=85)

    p.add_argument("--dr_min", type=float, default=0.001)
    p.add_argument("--dr_max", type=float, default=0.05)
    p.add_argument("--cr_min", type=float, default=0.001)
    p.add_argument("--cr_max", type=float, default=1.0)

    p.add_argument("--interp", choices=["nearest", "linear", "spline"],
                   default="spline", help="Interpolation for tract overlays")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print fsleyes command and any stderr output")

    return p.parse_args()


def global_scene_block(outfile: str | None, xzoom: int, yzoom: int, zzoom: int,
                       world_xyz: tuple[float, float, float], hide_axes: str) -> str:
    base = [f'"{FSLBIN / "fsleyes"}"']
    if outfile:
        base += ["render", f'--outfile "{outfile}"', "--size 1800 800"]

    base += [
        "--scene ortho",
        f"--xzoom {xzoom} --yzoom {yzoom} --zzoom {zzoom}",
        f"--worldLoc {world_xyz[0]} {world_xyz[1]} {world_xyz[2]}",
        "--showLocation no",
        "--layout horizontal",
        "--hideCursor",
        "--bgColour 0.0 0.0 0.0",
        "--fgColour 1.0 1.0 1.0",
        hide_axes,
    ]
    return " ".join(base)


def tract_base_name(tract_id: str) -> str:
    return tract_id.replace("_l", "").replace("_r", "")


CMAPS = {
    "slf1": "green", "slf2": "red", "slf3": "blue",
    "cbd": "blue", "cbp": "pink", "cbt": "blue-lightblue",
    "cst": "hot", "fa": "cool", "str": "blue", "atr": "cool",
    "or": "blue", "vof": "cool",
    "fx": "copper", "uf": "blue-lightblue",
    "fma": "yellow", "fmi": "yellow", "mcp": "red"
}


def add_underlay(cmd: str, brain: str) -> str:
    cmd += (
        f' "{brain}"'
        f" --overlayType volume"
        f" --cmap greyscale"
        f" --displayRange 0.0 98%"
    )
    return cmd


def add_tracts(cmd: str, xtract_dir: Path, tracts: list[str], dr, cr, interp: str,
               use_mip: bool = True) -> str:
    """
    Append tract overlays.
    If use_mip is True → overlayType mip; otherwise overlayType volume.
    """
    dr_min, dr_max = dr
    cr_min, cr_max = cr

    for t in tracts:
        vol = xtract_dir / t / "densityNorm.nii.gz"
        if not vol.exists():
            sys.stderr.write(f"warning: missing tract volume: {vol}\n")
            continue

        base = tract_base_name(t)
        cmap = CMAPS.get(base, "red-yellow")

        cmd += f' "{vol}"'
        if use_mip:
            cmd += " --overlayType mip"
        else:
            cmd += " --overlayType volume"

        cmd += (
            f" --cmap {cmap}"
            f" -dr {dr_min} {dr_max}"
            f" -cr {cr_min} {cr_max}"
            f" -in {interp}"
            f' --name "{t}"'
        )
    return cmd


def run_view(
    brain: str,
    xtract_dir: Path,
    tracts: list[str],
    world_xyz: tuple[float, float, float],
    hide_axes: str,
    args,
    outfile: str | None = None,
    use_mip: bool = True,
):
    cmd = global_scene_block(outfile, args.xzoom, args.yzoom, args.zzoom, world_xyz, hide_axes)
    cmd = add_underlay(cmd, brain)
    cmd = add_tracts(cmd, xtract_dir, tracts, (args.dr_min, args.dr_max), (args.cr_min, args.cr_max),
                     args.interp, use_mip=use_mip)

    proc = run(cmd, verbose=args.verbose)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "error: fsleyes command failed.\n")
    elif outfile:
        print(f"Saved: {outfile}")


def mk_outfile(base_path: str | None, name: str) -> str | None:
    if base_path is None:
        return None
    p = Path(base_path)
    outdir = p.parent if p.suffix else p
    outdir.mkdir(parents=True, exist_ok=True)
    return (outdir / name).as_posix()


def main():
    args = parse_args()

    require_tool("fsleyes")
    require_tool("imtest")
    require_tool("fslstats")

    xtract_dir = Path(args.xtract)
    if not xtract_dir.is_dir():
        sys.stderr.write(f'error: XTRACT folder not found: "{xtract_dir}"\n')
        sys.exit(2)

    if not imtest(args.brain):
        sys.stderr.write(f'error: brain image not found or unreadable: "{args.brain}"\n')
        sys.exit(2)

    # Compute a single worldLoc (COG in mm) from the brain/brainmask
    world_xyz = get_cog_mm(args.brain)

    # 1) Sagittal (MIP)
    run_view(
        brain=args.brain,
        xtract_dir=xtract_dir,
        tracts=["slf1_r", "cbd_r", "cbp_r", "cst_r", "mcp", "vof_r", "fx_r", "cbt_r"],
        world_xyz=world_xyz,
        hide_axes="-yh -zh",
        args=args,
        outfile=mk_outfile(args.outfile, "xtract_sagittal.png"),
        use_mip=True,
    )

    # 2) Coronal (MIP)
    run_view(
        brain=args.brain,
        xtract_dir=xtract_dir,
        tracts=["cst_l", "cst_r", "fx_l", "fx_r", "fa_l", "fa_r", "str_l", "str_r"],
        world_xyz=world_xyz,
        hide_axes="-xh -zh",
        args=args,
        outfile=mk_outfile(args.outfile, "xtract_coronal.png"),
        use_mip=True,
    )

    # 3) Axial (MIP)
    run_view(
        brain=args.brain,
        xtract_dir=xtract_dir,
        tracts=["atr_l", "atr_r", "fma", "fmi", "or_l", "or_r", "uf_l", "uf_r"],
        world_xyz=world_xyz,
        hide_axes="-xh -yh",
        args=args,
        outfile=mk_outfile(args.outfile, "xtract_axial.png"),
        use_mip=True,
    )


if __name__ == "__main__":
    main()
