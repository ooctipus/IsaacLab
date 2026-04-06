#!/usr/bin/env python3
"""Behavioral tests for cluster/lib.py submission logic.

Tests verify that the exact LAUNCHING output matches expected patterns,
preserving the behavior observed in production terminal output.
"""

import pytest

from lib import (
    ParsedArgs,
    build_cluster_str,
    build_combos,
    derive_run_name,
    expand_preset_braces,
    parse_args,
)


# =============================================================================
# expand_preset_braces
# =============================================================================


class TestExpandPresetBraces:
    def test_no_braces(self):
        assert expand_preset_braces("peg_insert_12mm") == ["peg_insert_12mm"]

    def test_single_brace_group(self):
        result = expand_preset_braces("{a,b,c}")
        assert result == ["a", "b", "c"]

    def test_two_brace_groups(self):
        result = expand_preset_braces("{a,b},{x,y}")
        assert result == ["a,x", "a,y", "b,x", "b,y"]

    def test_brace_with_fixed_suffix(self):
        result = expand_preset_braces("{a,b},fixed")
        assert result == ["a,fixed", "b,fixed"]

    def test_brace_with_fixed_prefix(self):
        result = expand_preset_braces("pre_,{a,b}")
        assert result == ["pre_,a", "pre_,b"]

    def test_four_by_two(self):
        result = expand_preset_braces(
            "{peg_insert_12mm,peg_insert_16mm,peg_insert_8mm,peg_insert_4mm},{choice,accumulator}"
        )
        assert len(result) == 8
        assert "peg_insert_12mm,choice" in result
        assert "peg_insert_12mm,accumulator" in result
        assert "peg_insert_4mm,accumulator" in result

    def test_plain_comma_no_braces(self):
        assert expand_preset_braces("a,b,c") == ["a,b,c"]


# =============================================================================
# parse_args
# =============================================================================


class TestParseArgs:
    def test_cluster_args_absorbed(self):
        p = parse_args(["num_gpu=8", "memory=800", "--task=Foo"])
        assert p.cluster["num_gpu"] == "8"
        assert p.cluster["memory"] == "800"
        assert p.fixed["--task"] == "Foo"

    def test_cluster_args_with_dashes(self):
        p = parse_args(["--num_gpu=8", "--memory=800"])
        assert p.cluster["num_gpu"] == "8"
        assert p.cluster["memory"] == "800"

    def test_cluster_args_space_separated(self):
        p = parse_args(["--num_gpu", "8", "--memory", "800"])
        assert p.cluster["num_gpu"] == "8"
        assert p.cluster["memory"] == "800"

    @pytest.mark.parametrize("args", [
        ["pool=isaac-dev-l40s-04"],
        ["--pool=isaac-dev-l40s-04"],
        ["--pool", "isaac-dev-l40s-04"],
    ])
    def test_pool_resolution_all_forms(self, args):
        p = parse_args(args)
        assert p.pool == "isaac-dev-l40s-04"
        assert p.cluster["platform"] == "ovx-l40s"

    def test_unknown_pool_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["pool=nonexistent-pool"])

    def test_preset_no_braces_is_fixed(self):
        p = parse_args(["presets=peg_insert_12mm"])
        assert p.fixed["presets"] == "peg_insert_12mm"
        assert "presets" not in p.sweep

    def test_preset_with_braces_is_sweep(self):
        p = parse_args(["presets={a,b},{x,y}"])
        assert "presets" in p.sweep
        assert p.sweep["presets"] == ["a,x", "a,y", "b,x", "b,y"]

    def test_normal_comma_sweep(self):
        p = parse_args(["--seed=1,2,3"])
        assert "--seed" in p.sweep
        assert p.sweep["--seed"] == ["1", "2", "3"]

    def test_normal_comma_sweep_no_dashes(self):
        p = parse_args(["env.decimation=4,8"])
        assert "env.decimation" in p.sweep
        assert p.sweep["env.decimation"] == ["4", "8"]

    def test_bracketed_not_sweep(self):
        p = parse_args(["agent.hidden=[64,32]"])
        assert p.fixed["agent.hidden"] == "[64,32]"
        assert "agent.hidden" not in p.sweep

    def test_video_flag_bare(self):
        p = parse_args(["--video"])
        assert "--video" in p.cli_flags

    def test_video_flag_true(self):
        p = parse_args(["--video=True"])
        assert "--video" in p.cli_flags

    def test_video_flag_false(self):
        p = parse_args(["--video=False"])
        assert "--video" not in p.cli_flags

    def test_hydra_delete(self):
        p = parse_args(["~env.some.key"])
        assert "~env.some.key" in p.hydra_dels

    def test_run_name_captured(self):
        p = parse_args(["--run_name=myrun"])
        assert p.run_name == "myrun"
        assert "--run_name" not in p.fixed
        assert "run_name" not in p.fixed

    def test_bare_flag_passthrough(self):
        p = parse_args(["--distributed"])
        assert "--distributed" in p.cli_flags

    def test_platform_override(self):
        p = parse_args(["platform=gb200"])
        assert p.cluster["platform"] == "gb200"


# =============================================================================
# build_combos
# =============================================================================


class TestBuildCombos:
    def test_no_sweep(self):
        assert build_combos({}) == [{}]

    def test_single_sweep(self):
        combos = build_combos({"--seed": ["1", "2", "3"]})
        assert len(combos) == 3
        assert {"--seed": "1"} in combos

    def test_cross_product(self):
        combos = build_combos({
            "presets": ["a,x", "a,y", "b,x", "b,y"],
            "--seed": ["1", "2"],
        })
        assert len(combos) == 8

    def test_two_normal_sweeps(self):
        combos = build_combos({
            "--seed": ["1", "2"],
            "env.decimation": ["4", "8"],
        })
        assert len(combos) == 4


# =============================================================================
# derive_run_name
# =============================================================================


class TestDeriveRunName:
    def test_combo_only(self):
        assert derive_run_name("", {"presets": "a,x"}) == "a,x"

    def test_base_only(self):
        assert derive_run_name("myrun", {}) == "myrun"

    def test_base_and_combo(self):
        assert derive_run_name("myrun", {"presets": "a,x"}) == "myrun,a,x"

    def test_empty(self):
        assert derive_run_name("", {}) == ""

    def test_multi_sweep_values(self):
        name = derive_run_name("", {"presets": "a,x", "--seed": "42"})
        assert name == "a,x,42"


# =============================================================================
# build_cluster_str
# =============================================================================


class TestBuildClusterStr:
    def test_default_cluster(self):
        from lib import CLUSTER_DEFAULTS
        s = build_cluster_str(dict(CLUSTER_DEFAULTS))
        assert "image=factory" in s
        assert "num_gpu=1" in s
        assert "platform=dgx-h100" in s

    def test_overridden_cluster(self):
        c = dict(
            image="factory", num_gpu="8", num_cpu="100", memory="800",
            platform="ovx-l40s", dataset="isaac-lab-ppo-model",
            num_node="1", storage="512", master_port="29400",
        )
        s = build_cluster_str(c)
        assert "num_gpu=8" in s
        assert "memory=800" in s
        assert "platform=ovx-l40s" in s


# =============================================================================
# Integration: full LAUNCHING line from terminal output
# =============================================================================


class TestIntegrationLaunchingOutput:
    """Verify the exact LAUNCHING output matches production behavior."""

    def _get_launches(self, raw_args: list[str], script: str = "train.py") -> list[str]:
        """Simulate do_submit and capture LAUNCHING lines."""
        import io
        from contextlib import redirect_stdout
        from lib import build_fixed_str, launch_job

        p = parse_args(raw_args)
        cluster_str = build_cluster_str(p.cluster)
        fixed_str = build_fixed_str(p.fixed)
        combos = build_combos(p.sweep)

        lines = []
        for combo in combos:
            run_name = derive_run_name(p.run_name, combo)
            buf = io.StringIO()
            with redirect_stdout(buf):
                launch_job(script, p.pool, cluster_str, fixed_str, combo,
                           run_name, p.hydra_dels, p.cli_flags, dry_run=True)
            lines.append(buf.getvalue().strip())
        return lines

    @staticmethod
    def _assert_pool_platform_consistent(line: str):
        """Verify that --pool and platform= in a LAUNCHING line are consistent."""
        import re
        from lib import POOL_TO_PLATFORM
        pool_match = re.search(r"--pool (\S+)", line)
        platform_match = re.search(r"platform=(\S+)", line)
        if pool_match and platform_match:
            pool = pool_match.group(1)
            platform = platform_match.group(1)
            if pool in POOL_TO_PLATFORM:
                expected = POOL_TO_PLATFORM[pool]
                assert platform == expected, (
                    f"Pool '{pool}' should map to platform '{expected}', got '{platform}'"
                )

    def test_cartesian_preset_braces(self):
        launches = self._get_launches([
            "image=factory", "pool=isaac-dev-l40s-04",
            "--task=Isaac-Factory-Franka-JointPos-v0",
            "presets={peg_insert_12mm,peg_insert_16mm,peg_insert_8mm,peg_insert_4mm},{choice,accumulator}",
            "--num_envs=10240", "num_gpu=8", "num_cpu=100", "memory=800", "storage=512",
            "--logger=wandb", "--log_project_name=factory_manager1",
        ])
        assert len(launches) == 8

        # Verify first and last
        assert "presets=peg_insert_12mm,choice" in launches[0]
        assert "--run_name=peg_insert_12mm,choice" in launches[0]
        assert "presets=peg_insert_4mm,accumulator" in launches[-1]
        assert "--run_name=peg_insert_4mm,accumulator" in launches[-1]

        for line in launches:
            assert "num_gpu=8" in line
            assert "memory=800" in line
            assert "--task=Isaac-Factory-Franka-JointPos-v0" in line
            self._assert_pool_platform_consistent(line)

    def test_single_preset_no_sweep(self):
        launches = self._get_launches([
            "presets=peg_insert_12mm", "--task=Foo",
        ])
        assert len(launches) == 1
        assert "presets=peg_insert_12mm" in launches[0]

    def test_normal_comma_sweep(self):
        launches = self._get_launches([
            "--seed=1,2,3", "--task=Foo",
        ])
        assert len(launches) == 3
        assert "--seed=1" in launches[0]
        assert "--seed=2" in launches[1]
        assert "--seed=3" in launches[2]

    def test_cross_product_preset_and_seed(self):
        launches = self._get_launches([
            "presets={a,b},{x,y}", "--seed=1,2",
        ])
        assert len(launches) == 8

    def test_mixed_fixed_brace_preset(self):
        launches = self._get_launches([
            "presets={a,b},fixed",
        ])
        assert len(launches) == 2
        assert "presets=a,fixed" in launches[0]
        assert "presets=b,fixed" in launches[1]

    def test_args_is_single_token_in_cmd(self):
        """The args=... portion must be a single element in the cmd list,
        not split by spaces. Otherwise osmo treats script args as its own flags."""
        from lib import build_cmd

        p = parse_args([
            "pool=isaac-dev-l40s-04",
            "--task=Isaac-Factory", "--num_envs=10240",
            "presets=peg_insert_12mm",
        ])
        cluster_str = build_cluster_str(p.cluster)
        from lib import build_fixed_str
        fixed_str = build_fixed_str(p.fixed)
        combos = build_combos(p.sweep)
        combo = combos[0]
        run_name = derive_run_name(p.run_name, combo)
        cmd = build_cmd("train.py", p.pool, cluster_str, fixed_str, combo,
                        run_name, p.hydra_dels, p.cli_flags)

        args_elements = [e for e in cmd if e.startswith("args=")]
        assert len(args_elements) == 1, f"Expected exactly one args= element, got {args_elements}"
        args_val = args_elements[0]
        assert "--task=Isaac-Factory" in args_val
        assert "--num_envs=10240" in args_val
        assert "presets=peg_insert_12mm" in args_val

    @pytest.mark.parametrize("pool,expected_platform", [
        ("isaac-dev-h100-01", "dgx-h100"),
        ("isaac-dev-l40s-04", "ovx-l40s"),
        ("isaac-dex-l40s-04", "ovx-l40s"),
        ("groot-gb200-02", "gb200"),
        ("isaac-dev-l40-03", "ovx-l40"),
    ])
    def test_pool_to_platform_in_output(self, pool, expected_platform):
        launches = self._get_launches([f"pool={pool}", "--task=Foo"])
        assert len(launches) == 1
        self._assert_pool_platform_consistent(launches[0])
        assert f"platform={expected_platform}" in launches[0]
        assert f"--pool {pool}" in launches[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
