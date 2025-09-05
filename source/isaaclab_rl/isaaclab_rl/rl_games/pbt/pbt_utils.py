import os
import random
from isaaclab.utils import configclass
from collections import OrderedDict


@configclass
class DistributedArgs:
    distributed: bool = True
    nnodes: int = 1
    nproc_per_node: int = 1
    rank: int = -1
    master_port: str | None = None

    def get_args_list(self) -> list[str]:
        args = ['-m', 'torch.distributed.run',
                f"--nnodes={self.nnodes}",
                f"--nproc_per_node={self.nproc_per_node}"]
        if self.master_port:
            args.append(f"--master_port={self.master_port}")
        return args


@configclass
class EnvArgs:
    task: str | None = None
    seed: int = -1
    headless: bool = True
    num_envs: int | None = None

    def get_args_list(self) -> list[str]:
        list = []
        list.append(f"--task={self.task}")
        list.append(f"--seed={self.seed}")
        list.append(f"--num_envs={self.num_envs}")
        if self.headless:
            list.append("--headless")
        return list


@configclass
class RenderingArgs:
    camera_enabled: bool = False
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000

    def get_args_list(self) -> list[str]:
        args = []
        if self.camera_enabled:
            args.append("--enable_cameras")
        if self.video:
            args.extend(["--video", f"--video_length={self.video_length}", f"--video_interval={self.video_interval}"])
        return args


@configclass
class WandbArgs:
    enabled: bool = False
    project_name: str | None = None
    entity: str | None = None
    name: str | None = None

    def get_args_list(self) -> list[str]:
        args = []
        if self.enabled:
            args.append("--track")
            if self.entity:
                args.append(f"--wandb-entity={self.entity}")
            else:
                raise ValueError("entity must be specified if wandb is enabled")
            if self.project_name:
                args.append(f"--wandb-project-name={self.project_name}")
            if self.name:
                args.append(f"--wandb-name={self.name}")
        return args


def dump_env_sizes():
    # number of env vars
    n = len(os.environ)
    # total bytes in "KEY=VAL\0" for all envp entries
    total = sum(len(k) + 1 + len(v) + 1 for k, v in os.environ.items())
    # find the 5 largest values
    biggest = sorted(os.environ.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]

    print(f"[ENV MONITOR] vars={n}, total_bytes={total}")
    for k, v in biggest:
        print(f"    {k!r} length={len(v)} → {v[:60]}{'…' if len(v) > 60 else ''}")

    try:
        argmax = os.sysconf('SC_ARG_MAX')
        print(f"[ENV MONITOR] SC_ARG_MAX = {argmax}")
    except (ValueError, AttributeError):
        pass


def flatten_dict(d, prefix='', separator='.'):
    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value

    return res


def find_free_port(max_tries: int = 20) -> int:
    """
    Return an OS-allocated free TCP port.
    Retries a few times to avoid rare 'bad file descriptor' races.
    """
    import socket
    for _ in range(max_tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Let the kernel pick an available port.
            s.bind(("", 0))
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError:          # includes 'Bad file descriptor'
            s.close()
            continue
    # Fallback: choose a random high port (still extremely unlikely to collide)
    return random.randint(20000, 65000)


def filter_params(params, params_to_mutate):
    filtered_params = dict()

    for key, value in params.items():
        if key in params_to_mutate:
            if isinstance(value, str):
                try:
                    # trying to convert values such as "1e-4" to floats because yaml fails to recognize them as such
                    float_value = float(value)
                    value = float_value
                except ValueError:
                    pass
            filtered_params[key] = value
    return filtered_params
