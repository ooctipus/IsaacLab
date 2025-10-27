import itertools
import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Tuple


def _split_table_row(line: str) -> List[str]:
    # PrettyTable row like: | col1 | col2 | col3 |
    parts = [p.strip() for p in line.strip().split("|")]
    # split() yields leading/trailing empty entries due to borders; filter them
    return [p for p in parts if p]


def _clean_section_name(name: str) -> str:
    # remove child row prefixes like "  └─ " and strip
    return name.replace("└─", "").strip()


def parse_pretty_tables(text: str) -> List[Dict[str, Any]]:
    """Parse PrettyTable blocks from a blob of stdout text.

    Returns a list of dicts: {title: str, headers: [..], rows: [{col: val}], type: "per-step"|"epoch"}
    """
    lines = text.splitlines()
    tables: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Heuristic: title line is a single-column PrettyTable row
        if line.strip().startswith("|") and line.count("|") == 2:
            # Check surrounding borders
            if i - 1 >= 0 and i + 1 < len(lines) and lines[i - 1].strip().startswith("+") and lines[i + 1].strip().startswith("+"):
                title = _split_table_row(line)[0]
                # Header row starts after next two lines (border then header)
                if i + 3 < len(lines):
                    header_line = lines[i + 2]
                    header_border = lines[i + 3]
                    if header_line.strip().startswith("|") and header_border.strip().startswith("+"):
                        headers = _split_table_row(header_line)
                        # Data rows start after header_border
                        j = i + 4
                        rows: List[Dict[str, Any]] = []
                        while j < len(lines) and lines[j].strip().startswith("|"):
                            row_vals = _split_table_row(lines[j])
                            if len(row_vals) == len(headers):
                                rows.append(dict(zip(headers, row_vals)))
                            j += 1
                        # Classify table type
                        tbl_type = "per-step"
                        if "Avg ms/call" in headers:
                            tbl_type = "epoch"
                        tables.append({
                            "title": title,
                            "headers": headers,
                            "rows": rows,
                            "type": tbl_type,
                        })
                        i = j
                        continue
        i += 1
    return tables


def log_tables_to_wandb(wandb, tables: List[Dict[str, Any]], num_envs: int):
    """Logs parsed tables into W&B metrics.

    For per-step tables, logs `<title-ms>/<section> ms_per_step` and
    `<title-pct>/<section> pct` so they do not share the same card folder.
    For epoch-level tables, logs `<title>/<section> avg_ms_per_call`, `calls`, and `total_ms`.
    """
    for tbl in tables:
        title = tbl.get("title", "").strip()
        headers = tbl.get("headers", [])
        rows = tbl.get("rows", [])
        is_per_step = tbl.get("type") == "per-step"
        for row in rows:
            section = _clean_section_name(row.get("Section", "").strip())
            if is_per_step and "Avg ms/step" in headers and "% of step" in headers:
                try:
                    ms = float(row["Avg ms/step"].replace(",", ""))
                except Exception:
                    ms = None
                try:
                    pct = float(row["% of step"].replace(",", "").replace("%", ""))
                except Exception:
                    pct = None
                # Separate prefixes for ms vs pct so they render under
                # distinct cards in W&B.
                ms_prefix = f"{title}-ms/{section}" if section else f"{title}-ms"
                pct_prefix = f"{title}-percentage/{section}" if section else f"{title}-percentage/"
                log_dict = {}
                if ms is not None:
                    log_dict[f"{ms_prefix}"] = ms
                if pct is not None:
                    log_dict[f"{pct_prefix}"] = pct
                if log_dict:
                    # also log num_envs with every datapoint for easy grouping/axes
                    log_dict["num_envs"] = num_envs
                    wandb.log(log_dict)


def run_one(task_name: str, camera_size: str, camera_type: str, obj: str, rendering_mode: str,
            num_envs: int, num_steps: int, python_exe: str = sys.executable,
            project: str = "isaaclab-ablation") -> Tuple[int, str, str]:
    task = f"{task_name}-v0"
    hydra_overrides = [
        f"env.scene={camera_size}{camera_type}",
        f"env.scene.object={obj}",
    ]
    cmd = [
        python_exe,
        os.path.join("scripts", "benchmarks", "benchmark_rsl_rl.py"),
        "--task", task,
        "--num_envs", str(num_envs),
        "--max_iterations", str(num_steps),
        "--rendering_mode", rendering_mode,
        "--headless",
        "--enable_cameras",
        "--merge_runner_step"
        # keep separate tables by default; rely on script defaults for term-level
    ] + hydra_overrides

    # Initialize W&B run
    try:
        import wandb  # type: ignore
    except Exception:
        wandb = None

    run = None
    num_camera = "Duo_Camera" if "Duo" in task else "Single_Camera"
    run_name = f"{num_camera}/{rendering_mode}/{camera_size}/{camera_type}"
    if wandb is not None:
        run = wandb.init(project=project, name=run_name, config={
            "task": task,
            "camera_size": camera_size,
            "camera_type": camera_type,
            "object": obj,
            "rendering_mode": rendering_mode,
            "num_envs": num_envs,
            "max_iterations": num_steps,
            "cmd": " ".join(shlex.quote(c) for c in cmd),
        })

    # Execute benchmark and capture output
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Parse and log tables
    if wandb is not None and proc.stdout:
        tables = parse_pretty_tables(proc.stdout)
        log_tables_to_wandb(wandb, tables, num_envs)

    # Log overall status
    if wandb is not None:
        wandb.log({"returncode": proc.returncode})
        if run is not None:
            run.finish()

    return proc.returncode, proc.stdout, proc.stderr


def main():
    # Parameter grids
    task_names = [
        "Dexsuite-Kuka-Allegro-Lift-Single-Camera",
        "Dexsuite-Kuka-Allegro-Lift-Duo-Camera",
    ]
    camera_sizes = ["64x64", "128x128", "256x256"]
    camera_types = ["raycaster", "tiled_depth", "tiled_rgb", "tiled_albedo"]
    objects = ["default", "geometry", "unidex100"]
    rendering_modes = ["balanced", "performance"]
    num_envs_list = [32, 64, 128, 256]
    num_steps_list = [10]

    # Optional overrides via environment (to avoid massive runs by default)
    project = os.environ.get("WANDB_PROJECT", "isaaclab-ablation")
    limit = int(os.environ.get("ABLATE_LIMIT", "0"))  # 0 means no limit

    combos = list(itertools.product(task_names, camera_sizes, camera_types, objects, rendering_modes, num_envs_list, num_steps_list))
    if limit > 0:
        combos = combos[:limit]

    for idx, (task_name, cam_size, cam_type, obj, rendering_mode, num_envs, num_steps) in enumerate(combos, 1):
        print(f"[{idx}/{len(combos)}] Running: task={task_name} mode={rendering_mode} size={cam_size} type={cam_type} obj={obj} envs={num_envs} steps={num_steps}")
        rc, out, err = run_one(task_name, cam_size, cam_type, obj, rendering_mode, num_envs, num_steps)
        if rc != 0:
            sys.stderr.write(f"Command failed with rc={rc}. Stderr:\n{err}\n")


if __name__ == "__main__":
    main()
