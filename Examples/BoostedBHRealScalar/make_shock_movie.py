"""Movie of the shock formation from the plot files of a BoostedBH run.

usage:
    python make_shock_movie.py RUNDIR/params.txt [--out movie.mp4] [--fps 8]
                               [--every N] [--frames-dir DIR] [--view near|full]
    python make_shock_movie.py --from-frames DIR [--out movie.mp4] [--fps 8]

Every frame shows the same rho/rho0 and Mach maps as analyse_shock.py (z = 0
plane, BH frame using the BH velocity at that time): --view near (default) the
+-3.5 r_sg maps_<step>.png, --view full the whole-domain level-0
maps_full_<step>.png, which is where the Mach cone and the Bogoliubov ripples
of a dispersive (small mu M) run show up. The frames are
encoded as an H.264 mp4 (yuv420p, plays in QuickTime and browsers) with the
ffmpeg binary that comes with imageio-ffmpeg (pip install imageio-ffmpeg).

Without imageio-ffmpeg (e.g. on MN5, whose ffmpeg has no H.264 encoder) give
--frames-dir to only write PNG frames, then copy them to a machine that has
imageio-ffmpeg and run --from-frames there.
"""
import argparse
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_shock import (  # noqa: E402
    Run,
    full_map_figure,
    level0_grid,
    map_figure,
    read_layer,
)

DPI = 100  # map_figure is 13 x 5.6 inches -> 1300 x 560 pixels


def open_writer(out, size, fps):
    """H.264 writer from imageio-ffmpeg, or None if it is not installed"""
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    writer = imageio_ffmpeg.write_frames(
        out,
        size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=None,
        macro_block_size=2,
        output_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart"],
    )
    writer.send(None)  # start the ffmpeg process
    return writer


def figure_rgb(fig):
    fig.set_dpi(DPI)
    fig.canvas.draw()
    return np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])


def frames_from_run(params, every, view="near"):
    """yield (step, RGB image) for every N-th plot file of the run"""
    run = Run(params)
    files = sorted(
        glob.glob(os.path.join(run.hdf5_dir, run.plot_prefix + "*.3d.hdf5"))
    )[::every]
    if not files:
        raise SystemExit(f"no plot files {run.plot_prefix}*.3d.hdf5 in {run.hdf5_dir}")
    for n, fn in enumerate(files):
        step = int(fn.split("_")[-1].split(".")[0])
        t, patches, names, x_extent = read_layer(fn)
        if run.field is None:
            run.set_field("complex" if "phi_Re" in names else "real")
        xc = run.num("center", 0.5 * x_extent)
        v = float(run.v_bh(t))
        if view == "full":
            x, y, R, M = level0_grid(run, patches, xc, v)
            fig = full_map_figure(
                run, x, y, R, M, v,
                f"{run.field}  t = {t:.0f} M, v_BH = {v:.4f}, "
                f"M0 = {v / run.cs0:.2f}, level 0",
            )
        else:
            fig = map_figure(run, t, patches, xc, v)
        img = figure_rgb(fig)
        plt.close(fig)
        print(f"frame {n + 1}/{len(files)}: step {step}, t = {t:.0f}", flush=True)
        yield step, img


def frames_from_dir(frames_dir):
    """yield (step, RGB image) from maps_<step>.png files"""
    files = sorted(glob.glob(os.path.join(frames_dir, "maps_*.png")))
    if not files:
        raise SystemExit(f"no maps_*.png in {frames_dir}")
    for fn in files:
        img = mpimg.imread(fn)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
        step = int(os.path.basename(fn)[5:-4])
        yield step, np.ascontiguousarray(img[..., :3])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("params", nargs="?", help="params file of the run")
    ap.add_argument("--from-frames", help="encode existing maps_*.png frames")
    ap.add_argument("--out", default="shock_formation.mp4", help="output mp4")
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--every", type=int, default=1, help="use every N-th file")
    ap.add_argument("--frames-dir", help="also write the frames as PNG here")
    ap.add_argument(
        "--view",
        choices=("near", "full"),
        default="near",
        help="near: +-3.5 r_sg maps; full: whole domain from level 0",
    )
    args = ap.parse_args()
    if bool(args.params) == bool(args.from_frames):
        ap.error("give either a params file or --from-frames DIR")

    if args.from_frames:
        frames = frames_from_dir(args.from_frames)
    else:
        frames = frames_from_run(args.params, args.every, args.view)
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)

    writer = None
    for step, img in frames:
        if writer is None:
            writer = open_writer(args.out, (img.shape[1], img.shape[0]), args.fps)
            if writer is None and not args.frames_dir:
                raise SystemExit(
                    "imageio-ffmpeg is not installed: pip install imageio-ffmpeg, "
                    "or give --frames-dir to only write PNG frames"
                )
        if args.frames_dir:
            mpimg.imsave(os.path.join(args.frames_dir, f"maps_{step:06d}.png"), img)
        if writer:
            writer.send(img)
    if writer:
        writer.close()
        print("movie written to", args.out)
    else:
        print("frames written to", args.frames_dir, "(no movie: no imageio-ffmpeg)")


if __name__ == "__main__":
    main()
