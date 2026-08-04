# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .state_file import StateFile


class ContainerInterface:
    """A helper class for managing Isaac Lab containers."""

    def __init__(
        self,
        context_dir: Path,
        profile: str = "base",
        yamls: list[str] | None = None,
        envs: list[str] | None = None,
        statefile: StateFile | None = None,
        suffix: str | None = None,
    ):
        """Initialize the container interface with the given parameters.

        Args:
            context_dir:
                The context directory for Docker operations.
            profile:
                The profile name for the container. Defaults to "base".
            yamls:
                A list of yaml files to extend ``docker-compose.yaml`` settings. These are extended in the order
                they are provided. Defaults to None, in which case no additional yaml files are added.
            envs:
                A list of environment variable files to extend the ``.env.base`` file. These are extended in the order
                they are provided. Defaults to None, in which case no additional environment variable files are added.
            statefile:
                An instance of the :class:`Statefile` class to manage state variables. Defaults to None, in
                which case a new configuration object is created by reading the configuration file at the path
                ``context_dir/.container.cfg``.
            suffix:
                Optional docker image and container name suffix.  Defaults to None, in which case, the docker name
                suffix is set to the empty string. A hyphen is inserted in between the profile and the suffix if
                the suffix is a nonempty string.  For example, if "base" is passed to profile, and "custom" is
                passed to suffix, then the produced docker image and container will be named ``isaac-lab-base-custom``.
        """
        # set the context directory
        self.context_dir = context_dir

        # create a state-file if not provided
        # the state file is a manager of run-time state variables that are saved to a file
        if statefile is None:
            self.statefile = StateFile(path=self.context_dir / ".container.cfg")
        else:
            self.statefile = statefile

        # set the profile and container name
        self.profile = profile
        if self.profile == "isaaclab":
            # Silently correct from isaaclab to base, because isaaclab is a commonly passed arg
            # but not a real profile
            self.profile = "base"

        # set the docker image and container name suffix
        if suffix is None or suffix == "":
            # if no name suffix is given, default to the empty string as the name suffix
            self.suffix = ""
        else:
            # insert a hyphen before the suffix if a suffix is given
            self.suffix = f"-{suffix}"

        # set names for easier reference
        self.base_service_name = "isaac-lab-base"
        self.service_name = f"isaac-lab-{self.profile}"
        self.container_name = f"{self.service_name}{self.suffix}"
        self.image_name = f"{self.service_name}{self.suffix}:latest"

        # keep the environment variables from the current environment,
        # except make sure that the docker name suffix is set from the script
        self.environ = os.environ.copy()
        self.environ["DOCKER_NAME_SUFFIX"] = self.suffix

        # resolve the image extension through the passed yamls and envs
        self._resolve_image_extension(yamls, envs)
        self._add_wandb_credentials_mount()
        self._add_host_cache_mounts()
        # load the environment variables from the .env files
        self._parse_dot_vars()

    def print_info(self):
        """Print the container interface information."""
        print("=" * 60)
        print(f"{'DOCKER CONTAINER INFO':^60}")  # Centered title
        print("=" * 60)

        print(f"{'Profile:':25} {self.profile}")
        print(f"{'Suffix:':25} {self.suffix}")
        print(f"{'Service Name:':25} {self.service_name}")
        print(f"{'Image Name:':25} {self.image_name}")
        print(f"{'Container Name:':25} {self.container_name}")

        print("-" * 60)
        print(f"{'Docker Compose Arguments':^60}")
        print("-" * 60)
        print(f"{'YAMLs:':25} {' '.join(self.add_yamls)}")
        print(f"{'Profiles:':25} {' '.join(self.add_profiles)}")
        print(f"{'Env Files:':25} {' '.join(self.add_env_files)}")
        print("=" * 60)

    """
    Operations.
    """

    def is_container_running(self) -> bool:
        """Check if the container is running.

        Returns:
            True if the container is running, otherwise False.
        """
        status = subprocess.run(
            ["docker", "container", "inspect", "-f", "{{.State.Status}}", self.container_name],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return status == "running"

    def does_image_exist(self) -> bool:
        """Check if the Docker image exists.

        Returns:
            True if the image exists, otherwise False.
        """
        return self._does_image_exist(self.image_name)

    @property
    def build_flags(self) -> list[str]:
        """Extra flags appended to every ``docker compose build`` invocation.

        Setting ``ISAACLAB_NOCACHE`` to a truthy value disables the Docker layer cache, so a
        rebuild cannot silently reuse previously resolved dependency layers.
        """
        if os.environ.get("ISAACLAB_NOCACHE", "").strip().lower() in ("1", "true", "yes", "on"):
            return ["--no-cache"]
        return []

    def _run_build(self, cmd: list[str], service_name: str):
        """Run a build command and fail loudly when Docker reports an error.

        Args:
            cmd: The docker compose build command to run.
            service_name: The service being built, used in the error message.

        Raises:
            RuntimeError: If the build command exits with a non-zero status.
        """
        result = subprocess.run(cmd, check=False, cwd=self.context_dir, env=self.environ)
        if result.returncode != 0:
            raise RuntimeError(
                f"[ERROR] Failed to build the docker image for service '{service_name}'"
                f" (exit code {result.returncode}). See the Docker build output above."
            )

    def build(self):
        """Build the Docker image."""
        print("[INFO] Building the docker image for the profile 'base'...\n")
        # build the image for the base profile
        cmd = (
            ["docker", "compose"]
            + ["--file", "docker-compose.yaml"]
            + ["--profile", "base"]
            + ["--env-file", ".env.base"]
            + ["build", self.base_service_name]
            + self.build_flags
        )
        self._run_build(cmd, self.base_service_name)
        print("[INFO] Finished building the docker image for the profile 'base'.\n")

        # build the image for the profile
        if self.profile != "base":
            print(f"[INFO] Building the docker image for the profile '{self.profile}'...\n")
            cmd = (
                ["docker", "compose"]
                + self.add_yamls
                + self.add_profiles
                + self.add_env_files
                + ["build", self.service_name]
                + self.build_flags
            )
            self._run_build(cmd, self.service_name)
            print(f"[INFO] Finished building the docker image for the profile '{self.profile}'.\n")

    def start(self, build: bool = False):
        """Start the Docker container using the Docker compose command.

        Args:
            build: If True, build/rebuild the image before starting. If False, reuse existing images
                and only build when the requested image is missing.
        """
        print(f"[INFO] Starting the container '{self.container_name}' in the background...\n")
        # Check if the container history file exists
        container_history_file = self.context_dir / ".isaac-lab-docker-history"
        if not container_history_file.exists():
            # Create the file with sticky bit on the group
            container_history_file.touch(mode=0o2644, exist_ok=True)

        # build the image for the base profile if not running base and the base image is missing
        base_image_name = f"{self.base_service_name}{self.suffix}:latest"
        if self.profile != "base" and (build or not self._does_image_exist(base_image_name)):
            cmd = (
                ["docker", "compose"]
                + ["--file", "docker-compose.yaml"]
                + ["--profile", "base"]
                + ["--env-file", ".env.base"]
                + ["build", self.base_service_name]
                + self.build_flags
            )
            self._run_build(cmd, self.base_service_name)

        if build or not self.does_image_exist():
            cmd = (
                ["docker", "compose"]
                + self.add_yamls
                + self.add_profiles
                + self.add_env_files
                + ["build", self.service_name]
                + self.build_flags
            )
            if not build:
                print(f"[INFO] Docker image '{self.image_name}' does not exist. Building it once before start.\n")
            self._run_build(cmd, self.service_name)

        # start the container without forcing a rebuild
        cmd = (
            ["docker", "compose"]
            + self.add_yamls
            + self.add_profiles
            + self.add_env_files
            + ["up", "--detach", "--no-build", "--remove-orphans"]
        )
        subprocess.run(cmd, check=False, cwd=self.context_dir, env=self.environ)

    def enter(self):
        """Enter the running container by executing a bash shell.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print(f"[INFO] Entering the existing '{self.container_name}' container in a bash session...\n")
            cmd = (
                ["docker", "exec", "--interactive", "--tty"]
                + (["-e", f"DISPLAY={os.environ['DISPLAY']}"] if "DISPLAY" in os.environ else [])
                + [self.container_name, "bash"]
            )
            subprocess.run(cmd)
            self.fix_output_ownership()
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running.")

    def stop(self):
        """Stop the running container using the Docker compose command."""
        if self.is_container_running():
            print(f"[INFO] Stopping the launched docker container '{self.container_name}'...\n")
            self.fix_output_ownership()
            # stop running services
            cmd = (
                ["docker", "compose"] + self.add_yamls + self.add_profiles + self.add_env_files + ["down", "--volumes"]
            )
            subprocess.run(cmd, check=False, cwd=self.context_dir, env=self.environ)
        else:
            print(
                f"[INFO] Can't stop container '{self.container_name}' as it is not running."
                " To check if the container is running, run 'docker ps' or 'docker container ls'.\n"
            )

    def copy(self, output_dir: Path | None = None):
        """Copy artifacts from the running container to the host machine.

        Args:
            output_dir: The directory to copy the artifacts to. Defaults to None, in which case
                the context directory is used.

        Raises:
            RuntimeError: If the container is not running.
        """
        if self.is_container_running():
            print(f"[INFO] Copying artifacts from the '{self.container_name}' container...\n")
            if output_dir is None:
                output_dir = self.context_dir

            # create a directory to store the artifacts
            output_dir = output_dir.joinpath("artifacts")
            if not output_dir.is_dir():
                output_dir.mkdir()

            # define dictionary of mapping from docker container path to host machine path
            docker_isaac_lab_path = Path(self.dot_vars["DOCKER_ISAACLAB_PATH"])
            artifacts = {
                docker_isaac_lab_path.joinpath("logs"): output_dir.joinpath("logs"),
                docker_isaac_lab_path.joinpath("docs/_build"): output_dir.joinpath("docs"),
                docker_isaac_lab_path.joinpath("data_storage"): output_dir.joinpath("data_storage"),
            }
            # print the artifacts to be copied
            for container_path, host_path in artifacts.items():
                print(f"\t -{container_path} -> {host_path}")
            # remove the existing artifacts
            for path in artifacts.values():
                shutil.rmtree(path, ignore_errors=True)

            # copy the artifacts
            for container_path, host_path in artifacts.items():
                cmd = ["docker", "cp", f"{self.container_name}:{container_path}/", host_path]
                subprocess.run(cmd, check=False, cwd=self.context_dir, env=self.environ)
            print("\n[INFO] Finished copying the artifacts from the container.")
        else:
            raise RuntimeError(f"The container '{self.container_name}' is not running.")

    def fix_output_ownership(self):
        """Give bind-mounted output artifacts back to the host user.

        Isaac Sim containers run as root by default. This keeps that behavior, but prevents root-owned files under
        host-mounted output directories such as ``models_tmp`` after using ``container.py enter`` or ``stop``.
        """
        if not self.is_container_running():
            return

        docker_isaac_lab_path = Path(self.dot_vars["DOCKER_ISAACLAB_PATH"])
        output_paths = [docker_isaac_lab_path.joinpath("models_tmp")]
        uid = os.getuid()
        gid = os.getgid()

        for output_path in output_paths:
            print(f"[INFO] Fixing host ownership for container output path: {output_path}")
            cmd = [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-lc",
                f"if [ -e '{output_path}' ]; then chown -R {uid}:{gid} '{output_path}'; fi",
            ]
            subprocess.run(cmd, check=False, cwd=self.context_dir, env=self.environ)

    def config(self, output_yaml: Path | None = None):
        """Process the Docker compose configuration based on the passed yamls and environment files.

        If the :attr:`output_yaml` is not None, the configuration is written to the file. Otherwise, it is printed to
        the terminal.

        Args:
            output_yaml: The path to the yaml file where the configuration is written to. Defaults
                to None, in which case the configuration is printed to the terminal.
        """
        print("[INFO] Configuring the passed options into a yaml...\n")

        # resolve the output argument
        if output_yaml is not None:
            output = ["--output", output_yaml]
        else:
            output = []

        # run the docker compose config command to generate the configuration
        cmd = ["docker", "compose"] + self.add_yamls + self.add_profiles + self.add_env_files + ["config"] + output
        subprocess.run(cmd, check=False, cwd=self.context_dir, env=self.environ)

    """
    Helper functions.
    """

    def _does_image_exist(self, image_name: str) -> bool:
        """Check if a Docker image exists."""
        result = subprocess.run(["docker", "image", "inspect", image_name], capture_output=True, text=True)
        return result.returncode == 0

    def _add_wandb_credentials_mount(self) -> None:
        """Mount host W&B credentials when available without making them required for all users."""
        netrc_path = Path.home() / ".netrc"
        wandb_yaml = self.context_dir / "docker-compose.wandb.yaml"
        if netrc_path.is_file() and wandb_yaml.is_file():
            self.environ["HOST_WANDB_NETRC_PATH"] = str(netrc_path)
            self.add_yamls += ["--file", "docker-compose.wandb.yaml"]

    def _add_host_cache_mounts(self) -> None:
        """Use existing host shader/cache directories to avoid recompiling shaders after container recreation."""
        cache_mounts = [
            (
                "HOST_ISAACSIM_KIT_CACHE_PATH",
                self.context_dir.parent / "_isaac_sim" / "kit" / "cache",
                Path("isaac-sim/kit/cache"),
                "${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache",
            ),
            (
                "HOST_OMNIVERSE_CACHE_PATH",
                Path.home() / ".cache" / "ov",
                Path("ov"),
                "${DOCKER_USER_HOME}/.cache/ov",
            ),
            (
                "HOST_NVIDIA_GL_CACHE_PATH",
                Path.home() / ".cache" / "nvidia" / "GLCache",
                Path("nvidia/GLCache"),
                "${DOCKER_USER_HOME}/.cache/nvidia/GLCache",
            ),
            (
                "HOST_NVIDIA_COMPUTE_CACHE_PATH",
                Path.home() / ".nv" / "ComputeCache",
                Path("nv/ComputeCache"),
                "${DOCKER_USER_HOME}/.nv/ComputeCache",
            ),
            (
                "HOST_NVIDIA_OPTIX_CACHE_PATH",
                Path.home() / ".cache" / "NVIDIA" / "OptixCache",
                Path("NVIDIA/OptixCache"),
                "${DOCKER_USER_HOME}/.cache/NVIDIA/OptixCache",
            ),
        ]
        existing_cache_mounts = [
            (env_name, source, target)
            for env_name, default_source, shared_root_suffix, target in cache_mounts
            if (source := self._resolve_host_cache_path(env_name, default_source, shared_root_suffix)) is not None
        ]
        host_cache_yaml = self.context_dir / ".isaac-lab-host-cache.yaml"

        if not existing_cache_mounts:
            host_cache_yaml.unlink(missing_ok=True)
            return

        volume_lines = [
            f"      - type: bind\n        source: ${{{env_name}}}\n        target: {target}\n"
            for env_name, _source, target in existing_cache_mounts
        ]
        volumes_block = "".join(volume_lines)
        host_cache_yaml.write_text(
            "# Generated by docker/container.py. Do not commit.\n"
            "services:\n"
            "  isaac-lab-base:\n"
            "    volumes:\n"
            f"{volumes_block}"
            "\n"
            "  isaac-lab-ros2:\n"
            "    volumes:\n"
            f"{volumes_block}",
            encoding="utf-8",
        )
        self.add_yamls += ["--file", ".isaac-lab-host-cache.yaml"]

    def _resolve_host_cache_path(self, env_name: str, default_source: Path, shared_root_suffix: Path) -> Path | None:
        """Resolve a host cache path and export it for the generated compose override."""
        source = Path(self.environ[env_name]) if env_name in self.environ else None
        if source is None and "HOST_ISAACLAB_CACHE_ROOT" in self.environ:
            source = Path(self.environ["HOST_ISAACLAB_CACHE_ROOT"]) / shared_root_suffix
        if source is None:
            source = default_source

        if not source.is_dir():
            return None

        self.environ[env_name] = str(source)
        return source

    def _resolve_image_extension(self, yamls: list[str] | None = None, envs: list[str] | None = None):
        """Resolve the image extension by setting up YAML files, profiles, and environment files for the
        Docker compose command.

        Args:
            yamls: A list of yaml files to extend ``docker-compose.yaml`` settings. These are extended in the order
                they are provided.
            envs: A list of environment variable files to extend the ``.env.base`` file. These are extended in the order
                they are provided.
        """
        self.add_yamls = ["--file", "docker-compose.yaml"]
        self.add_profiles = ["--profile", f"{self.profile}"]
        self.add_env_files = ["--env-file", ".env.base"]

        # extend env file based on profile
        if self.profile != "base":
            self.add_env_files += ["--env-file", f".env.{self.profile}"]

        # extend the env file based on the passed envs
        if envs is not None:
            for env in envs:
                self.add_env_files += ["--env-file", env]

        # extend the docker-compose.yaml based on the passed yamls
        if yamls is not None:
            for yaml in yamls:
                self.add_yamls += ["--file", yaml]

    def _parse_dot_vars(self):
        """Parse the environment variables from the .env files.

        Based on the passed ".env" files, this function reads the environment variables and stores them in a dictionary.
        The environment variables are read in order and overwritten if there are name conflicts, mimicking the behavior
        of Docker compose.
        """
        self.dot_vars: dict[str, Any] = {}

        # check if the number of arguments is even for the env files
        if len(self.add_env_files) % 2 != 0:
            raise RuntimeError(
                "The parameters for env files are configured incorrectly. There should be an even number of arguments."
                f" Received: {self.add_env_files}."
            )

        # read the environment variables from the .env files
        for i in range(1, len(self.add_env_files), 2):
            with open(self.context_dir / self.add_env_files[i]) as f:
                self.dot_vars.update(dict(line.strip().split("=", 1) for line in f if "=" in line))
