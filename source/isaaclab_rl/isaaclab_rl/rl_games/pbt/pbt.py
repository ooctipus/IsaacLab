import os
import sys
import random
import shutil
from pathlib import Path
from typing import Dict, List
from pprint import pprint

import numpy as np
import yaml

from rl_games.algos_torch.torch_ext import safe_save, safe_filesystem_op
from rl_games.common.algo_observer import AlgoObserver
from .mutation import mutate

import torch
import datetime
import torch.distributed as dist
from .pbt_cfg import PbtCfg
from . import pbt_utils

# i.e. value for target objective when it is not known
_UNINITIALIZED_VALUE = float(-1e9)


class PbtAlgoObserver(AlgoObserver):
    def __init__(self, params, args_cli):
        super().__init__()

        self.dir = params["pbt"]["directory"]

        self.rendering_args = pbt_utils.RenderingArgs(
            camera_enabled=args_cli.enable_cameras,
            video=args_cli.video,
            video_length=args_cli.video_length,
            video_interval=args_cli.video_interval
        )
        self.wandb_args = pbt_utils.WandbArgs(
            enabled=args_cli.track,
            project_name=args_cli.wandb_project_name,
            name=args_cli.wandb_name,
            entity=args_cli.wandb_entity
        )
        self.env_args = pbt_utils.EnvArgs(
            task=args_cli.task,
            seed=args_cli.seed if args_cli.seed is not None else -1,
            headless=args_cli.headless,
            num_envs=args_cli.num_envs
        )
        self.distributed_args = pbt_utils.DistributedArgs(
            distributed=args_cli.distributed,
            nproc_per_node=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0))
        )
        self.cfg = PbtCfg(**params['pbt'])
        self.pbt_iteration = -1  # dummy value, stands for "not initialized"
        self.curr_target_objective_value = _UNINITIALIZED_VALUE
        self.params = pbt_utils.filter_params(pbt_utils.flatten_dict({"agent": params}), self.cfg.mutation)

        assert len(self.params) > 0, "[DANGER]: Dictionary that contains params to mutate is empty"
        print('----------------------------------------')
        print(f'List of params to mutate: {self.params=}')

        self.device = params["params"]["config"]["device"]
        self.restart_flag = torch.tensor([0], device=self.device)

    def after_init(self, algo):
        if self.distributed_args.rank != 0:
            return

        self.algo = algo
        self.root_dir = algo.train_dir
        self.curr_policy_workspace_dir = os.path.join(os.path.join(self.root_dir, self.cfg.workspace), f'{self.cfg.policy_idx:03d}')
        os.makedirs(self.curr_policy_workspace_dir, exist_ok=True)

    def process_infos(self, infos, done_indices):
        if "true_objective" not in infos and 'Curriculum/adr' in infos['episode']:
            # case for curriculum
            if isinstance(infos['episode']['Curriculum/adr'], torch.Tensor):
                infos['true_objective'] = infos['episode']['Curriculum/adr'].float().mean().item()
            else:
                infos['true_objective'] = infos['episode']['Curriculum/adr']
        elif "true_objective" in infos and isinstance(infos["true_objective"], torch.Tensor):
            # case for direct
            infos['true_objective'] = infos["true_objective"].float().mean().item()

        if 'true_objective' in infos and isinstance(infos['true_objective'], (int, float)):
            self.curr_target_objective_value = infos['true_objective']
        else:
            if self.algo.game_rewards.current_size >= self.algo.games_to_track:
                self.curr_target_objective_value = float(self.algo.mean_rewards)

    def after_steps(self):
        if self.distributed_args.rank != 0:
            if self.restart_flag.cpu().item() == 1:
                print('Exiting this process on device = {}'.format(self.device))
                os._exit(0)
            return

        elif self.restart_flag.cpu().item() == 1:
            print(f'Restarting the process with new params on {self.distributed_args.distributed=}, {self.device=}')
            self._restart_with_new_params(self.restart_params['new_params'], self.restart_params['restart_from_checkpoint'])
            return

        if self.algo.frame // self.cfg.interval_steps <= self.pbt_iteration:
            return

        self.pbt_iteration = self.algo.frame // self.cfg.interval_steps
        frame_left = (self.pbt_iteration + 1) * self.cfg.interval_steps - self.algo.frame
        print(f'Policy {self.cfg.policy_idx}, frames_left {frame_left}, PBT it {self.pbt_iteration}')

        try:
            self._save_pbt_checkpoint()
            ckpts = self._load_population_checkpoints()
            self._cleanup(ckpts)
        except Exception as exc:
            print(f'Policy {self.cfg.policy_idx}: Exception {exc} during sanity log!')
            return

        sumry = {i: (None if c is None else {k: v for k, v in c.items() if k != 'params'}) for i, c in ckpts.items()}

        print(f"Current policy {self.cfg.policy_idx}, checkpoints (params hidden):")
        pprint(sumry, width=120)

        policies = list(range(self.cfg.num_policies))
        target_objectives = [ckpts[p]['true_objective'] if ckpts[p] else _UNINITIALIZED_VALUE for p in policies]
        initialized = [(obj, p) for obj, p in zip(target_objectives, policies) if obj > _UNINITIALIZED_VALUE]
        if not initialized:
            print("No policies initialized; skipping PBT iteration.")
            return
        initialized_objectives, initialized_policies = zip(*initialized)

        # 1) Stats
        mean_obj = float(np.mean(initialized_objectives))
        std_obj = float(np.std(initialized_objectives))
        upper_cut = mean_obj + self.cfg.replace_threshold_frac_std * std_obj
        lower_cut = mean_obj - self.cfg.replace_threshold_frac_std * std_obj

        # 2) Leaders & laggards
        leaders = [p for obj, p in zip(initialized_objectives, initialized_policies) if obj > upper_cut and (obj - mean_obj) > self.cfg.replace_threshold_frac_absolute]
        laggards = [p for obj, p in zip(initialized_objectives, initialized_policies) if obj < lower_cut and (max(initialized_objectives) - obj) > self.cfg.replace_threshold_frac_absolute]

        print(f"mean={mean_obj:.4f}, std={std_obj:.4f}, upper={upper_cut:.4f}, lower={lower_cut:.4f}")
        print(f"Leaders: {leaders}")
        print(f"Laggards: {laggards}")

        # 3) Best‐policy summary
        best_policy = max(zip(initialized_objectives, initialized_policies), key=lambda x: x[0])[1]
        best_objective = max(initialized_objectives)
        self._maybe_save_best_policy(best_objective, best_policy, ckpts[best_policy])

        # 4) Only replace if *this* policy is a laggard
        # if self.cfg.policy_idx not in laggards:
        #     print(f"Policy {self.cfg.policy_idx} is within the normal band; no weight replacement.")
        #     return

        # 5) If there are any leaders, pick one at random; else simply mutate with no replacement
        if leaders:
            replacement_policy_candidate = random.choice(leaders)
            print(f"Replacing policy {self.cfg.policy_idx} with random leader {replacement_policy_candidate}.")
        else:
            print("No leader exceeds thresholds; mutating in place.")
            replacement_policy_candidate = self.cfg.policy_idx

        self._pbt_summaries(self.params, best_objective)
        # Decided to replace the policy weights!
        new_params = ckpts[replacement_policy_candidate]['params']
        new_params = mutate(new_params, self.cfg.mutation, self.cfg.mutation_rate, self.cfg.change_min, self.cfg.change_max)

        restart_from_checkpoint = os.path.abspath(ckpts[replacement_policy_candidate]['checkpoint'])
        experiment_name = ckpts[self.cfg.policy_idx]['experiment_name']

        print(f'Policy {self.cfg.policy_idx}: Preparing to restart the process with mutated parameters!')
        self.restart_flag[0] = 1
        if self.distributed_args.distributed:
            dist.broadcast(self.restart_flag, src=0)

        self.restart_params = dict()
        self.restart_params['new_params'] = new_params
        self.restart_params['restart_from_checkpoint'] = restart_from_checkpoint
        self.restart_params['experiment_name'] = experiment_name

    def _save_pbt_checkpoint(self):
        if self.distributed_args.rank != 0:
            return

        """Save PBT-specific information including iteration number, policy index and hyperparameters."""
        checkpoint_file = os.path.join(self.curr_policy_workspace_dir, f'{self.pbt_iteration:06d}.pth')
        algo_state = self.algo.get_full_state_weights()
        safe_save(algo_state, checkpoint_file)

        pbt_checkpoint_file = os.path.join(self.curr_policy_workspace_dir, f'{self.pbt_iteration:06d}.yaml')

        pbt_checkpoint = {
            'iteration': self.pbt_iteration,
            'true_objective': self.curr_target_objective_value,
            'frame': self.algo.frame,
            'params': self.params,
            'checkpoint': os.path.abspath(checkpoint_file),
            'pbt_checkpoint': os.path.abspath(pbt_checkpoint_file),
            'experiment_name': self.algo.experiment_name,
        }

        print(f'Policy {self.cfg.policy_idx}: PBT checkpoint saving the dict {pbt_checkpoint} in {pbt_checkpoint_file} ...')
        with open(pbt_checkpoint_file, 'w') as fobj:
            print(f'Policy {self.cfg.policy_idx}: Saving {pbt_checkpoint_file}...')
            yaml.dump(pbt_checkpoint, fobj)

    def _load_population_checkpoints(self):
        if self.distributed_args.rank != 0:
            return

        """
        Load checkpoints for other policies in the population.
        Pick the newest checkpoint, but not newer than our current iteration.
        """
        checkpoints = dict()

        for policy_idx in range(self.cfg.num_policies):
            checkpoints[policy_idx] = None
            policy_dir = os.path.join(os.path.join(self.root_dir, self.cfg.workspace), f'{policy_idx:03d}')

            if not os.path.isdir(policy_dir):
                print(f'Policy {self.cfg.policy_idx}: {policy_idx} does not exist in {policy_dir}')
                continue

            pbt_checkpoint_files = [f for f in os.listdir(policy_dir) if f.endswith('.yaml')]
            pbt_checkpoint_files.sort(reverse=True)

            for pbt_checkpoint_file in pbt_checkpoint_files:
                iteration_str = pbt_checkpoint_file.split('.')[0]
                iteration = int(iteration_str)

                # current local time
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ctime_ts = os.path.getctime(os.path.join(policy_dir, pbt_checkpoint_file))
                created_str = datetime.datetime.fromtimestamp(ctime_ts).strftime("%Y-%m-%d %H:%M:%S")   

                if iteration <= self.pbt_iteration:
                    with open(os.path.join(policy_dir, pbt_checkpoint_file), 'r') as fobj:
                        print(f'Policy {self.cfg.policy_idx} [{now_str}]: Loading policy-{policy_idx} {pbt_checkpoint_file} (created at {created_str})')
                        checkpoints[policy_idx] = safe_filesystem_op(yaml.load, fobj, Loader=yaml.FullLoader)
                        break
                else:
                    print(f'Policy {self.cfg.policy_idx}: Not loading {pbt_checkpoint_file} \
                        because current {iteration} < {self.pbt_iteration}')                    
                    pass

        assert self.cfg.policy_idx in checkpoints.keys()
        return checkpoints

    def _maybe_save_best_policy(self, best_objective, best_policy_idx: int, best_policy_checkpoint):
        if self.distributed_args.rank != 0:
            return

        # make a directory containing best policy checkpoints using safe_filesystem_op
        best_policy_workspace_dir = os.path.join(os.path.join(self.root_dir, self.cfg.workspace), f'best{self.cfg.policy_idx}')
        safe_filesystem_op(os.makedirs, best_policy_workspace_dir, exist_ok=True)

        best_objective_so_far = _UNINITIALIZED_VALUE

        best_policy_checkpoint_files = [f for f in os.listdir(best_policy_workspace_dir) if f.endswith('.yaml')]
        best_policy_checkpoint_files.sort(reverse=True)
        if best_policy_checkpoint_files:
            with open(os.path.join(best_policy_workspace_dir, best_policy_checkpoint_files[0]), 'r') as fobj:
                best_policy_checkpoint_so_far = safe_filesystem_op(yaml.load, fobj, Loader=yaml.FullLoader)
                best_objective_so_far = best_policy_checkpoint_so_far['true_objective']

        if best_objective_so_far >= best_objective:
            # don't save the checkpoint if it is worse than the best checkpoint so far
            return

        print(f'Policy {self.cfg.policy_idx}: New best objective: {best_objective}!')

        # save the best policy checkpoint to this folder
        best_policy_checkpoint_name = f'{self.env_args.task}_best_obj_{best_objective:015.5f}_iter_{self.pbt_iteration:04d}_policy{best_policy_idx:03d}_frame{self.algo.frame}'

        # copy the checkpoint file to the best policy directory
        try:
            shutil.copy(best_policy_checkpoint['checkpoint'], os.path.join(best_policy_workspace_dir, f'{best_policy_checkpoint_name}.pth'))
            shutil.copy(best_policy_checkpoint['pbt_checkpoint'], os.path.join(best_policy_workspace_dir, f'{best_policy_checkpoint_name}.yaml'))

            # cleanup older best policy checkpoints, we want to keep only N latest files
            best_policy_checkpoint_files = [f for f in os.listdir(best_policy_workspace_dir)]
            best_policy_checkpoint_files.sort(reverse=True)

            n_to_keep = 6
            for best_policy_checkpoint_file in best_policy_checkpoint_files[n_to_keep:]:
                os.remove(os.path.join(best_policy_workspace_dir, best_policy_checkpoint_file))

        except Exception as exc:
            print(f'Policy {self.cfg.policy_idx}: Exception {exc} when copying best checkpoint!')
            # no big deal if this fails, hopefully the next time we will succeeed
            return

    def _pbt_summaries(self, params, best_objective):
        if self.distributed_args.rank == 0:
            for param, value in params.items():
                self.algo.writer.add_scalar(f'zz_pbt/{param}', value, self.algo.frame)
            self.algo.writer.add_scalar('zz_pbt/00_best_objective', best_objective, self.algo.frame)
            self.algo.writer.flush()

    def _cleanup(self, checkpoints: Dict[int, dict], keep_back: int = 20, max_yaml: int = 50) -> None:
        if self.distributed_args.rank == 0:
            oldest = min((ckpt['iteration'] if ckpt else 0) for ckpt in checkpoints.values())
            threshold = max(0, oldest - keep_back)
            root = Path(self.curr_policy_workspace_dir)

            # group files by numeric iteration (only *.yaml / *.pth)
            groups: Dict[int, List[Path]] = {}
            for p in root.iterdir():
                if p.suffix not in ('.yaml', '.pth'):
                    continue
                if p.stem.isdigit():
                    groups.setdefault(int(p.stem), []).append(p)

            removed = 0

            # 1) drop anything older than threshold
            for it in [i for i in groups if i <= threshold]:
                for p in groups[it]:
                    p.unlink(missing_ok=True)
                    removed += 1
                groups.pop(it, None)

            # 2) cap total YAML checkpoints: keep newest `max_yaml` iters
            yaml_iters = sorted((i for i, ps in groups.items() if any(p.suffix == '.yaml' for p in ps)), reverse=True)
            for it in yaml_iters[max_yaml:]:
                for p in groups.get(it, []):
                    p.unlink(missing_ok=True)
                    removed += 1
                groups.pop(it, None)

    def _delete_old_checkpoint(self, pbt_checkpoint_files: List[str]) -> bool:
        if self.distributed_args.rank != 0:
            return False

        """
        Delete the checkpoint that results in the smallest max gap between the remaining checkpoints.
        Do not delete any of the last N checkpoints.
        """
        pbt_checkpoint_files.sort()
        n_latest_to_keep = 20
        candidates = pbt_checkpoint_files[:-n_latest_to_keep]
        num_candidates = len(candidates)
        if num_candidates < 3:
            return False

        def _iter(f):
            return int(f.split('.')[0])

        best_gap = 1e9
        best_candidate = 1
        for i in range(1, num_candidates - 1):
            prev_iteration = _iter(candidates[i - 1])
            next_iteration = _iter(candidates[i + 1])

            # gap is we delete the ith candidate
            gap = next_iteration - prev_iteration
            if gap < best_gap:
                best_gap = gap
                best_candidate = i

        # delete the best candidate
        best_candidate_file = candidates[best_candidate]
        files_to_remove = [best_candidate_file, f'{_iter(best_candidate_file):06d}.pth']
        for file_to_remove in files_to_remove:
            print(f'Policy {self.cfg.policy_idx}: PBT cleanup old checkpoints, removing checkpoint {file_to_remove} (best gap {best_gap})')
            os.remove(os.path.join(self.curr_policy_workspace_dir, file_to_remove))

        return True

    def _restart_with_new_params(self, new_params, restart_from_checkpoint):

        cli_args = sys.argv
        print(f'previous command line args: {cli_args}')

        SKIP_KEYS = ['checkpoint']

        modified_args = [cli_args[0]]  # initialize with path to the Python script        
        for arg in cli_args[1:]:
            if '=' not in arg:
                modified_args.append(arg)
            else:
                assert '=' in arg
                arg_name, arg_value = arg.split('=')
                if arg_name in new_params or any(k in arg_name for k in SKIP_KEYS):
                    continue

                modified_args.append(f'{arg_name}={arg_value}')
        modified_args.append(f'--checkpoint={restart_from_checkpoint}')
        modified_args.extend(self.wandb_args.get_args_list())
        modified_args.extend(self.rendering_args.get_args_list())

        # add all of the new (possibly mutated) parameters
        for param, value in new_params.items():
            modified_args.append(f'{param}={value}')

        self.algo.writer.flush()
        self.algo.writer.close()

        if self.wandb_args.enabled:
            import wandb
            wandb.run.finish()

        # Get the directory of the current file
        thisfile_dir = os.path.dirname(os.path.abspath(__file__))
        isaac_sim_path = os.path.abspath(os.path.join(thisfile_dir, "../../../../_isaac_sim"))
        # ---------------------------------------------------------------------
        # Build the torch.distributed command
        # ---------------------------------------------------------------------
        command = [f'{isaac_sim_path}/python.sh']

        if self.distributed_args.distributed:
            master_port = pbt_utils.find_free_port()
            self.distributed_args.master_port = str(master_port)
            command.extend(self.distributed_args.get_args_list())
        command += [modified_args[0]]
        command.extend(self.env_args.get_args_list())
        command += modified_args[1:]
        if self.distributed_args.distributed:
            command += ['--distributed']

        print('Running command:', command, flush=True)
        print('sys.executable = ', sys.executable)
        print(f'Policy {self.cfg.policy_idx}: Restarting self with args {modified_args}', flush=True)

        if self.distributed_args.rank == 0:
            pbt_utils.dump_env_sizes()

            # after any sourcing (or before exec’ing python.sh) prevent kept increasing arg_length:
            for var in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS"):
                val = os.environ.get(var)
                if not val or os.pathsep not in val:
                    continue
                parts = val.split(os.pathsep)
                seen = set()
                new_parts = []
                for p in parts:
                    if p and p not in seen:
                        seen.add(p)
                        new_parts.append(p)
                os.environ[var] = os.pathsep.join(new_parts)

            os.execv(f'{isaac_sim_path}/python.sh', command)


class MultiObserver(AlgoObserver):
    """Meta-observer that allows the user to add several observers."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    def _call_multi(self, method, *args_, **kwargs_):
        for o in self.observers:
            getattr(o, method)(*args_, **kwargs_)

    def before_init(self, base_name, config, experiment_name):
        self._call_multi('before_init', base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi('after_init', algo)

    def process_infos(self, infos, done_indices):
        self._call_multi('process_infos', infos, done_indices)

    def after_steps(self):
        self._call_multi('after_steps')

    def after_clear_stats(self):
        self._call_multi('after_clear_stats')

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi('after_print_stats', frame, epoch_num, total_time)
