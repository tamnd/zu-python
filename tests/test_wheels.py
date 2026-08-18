"""The check the release job runs over what it built.

A gate that only runs on a tag is a gate nobody sees fail until the
release it fails, so the rule it enforces is written as a function and
the function is tested here. What the release job does with it is read
a directory and print the complaints.
"""

from __future__ import annotations

import wheel_tags

VERSION = "0.0.1"


def grid() -> list[str]:
    """A complete release: every ABI on every platform, and the sdist."""
    platforms = {
        "manylinux_2_28_x86_64": "manylinux_2_28_x86_64",
        "manylinux_2_28_aarch64": "manylinux_2_28_aarch64",
        "musllinux_1_2_x86_64": "musllinux_1_2_x86_64",
        "musllinux_1_2_aarch64": "musllinux_1_2_aarch64",
        # The tag set maturin writes for a universal2 build, which is
        # the wheel saying the three true things about itself and not
        # three wheels.
        "macosx_universal2": "macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2",
        "win_amd64": "win_amd64",
        "win_arm64": "win_arm64",
    }
    return [f"zudb-{VERSION}.tar.gz"] + [
        f"zudb-{VERSION}-{abi}-{platform}.whl"
        for abi in wheel_tags.ABIS
        for platform in platforms.values()
    ]


def test_a_complete_release_has_nothing_wrong_with_it() -> None:
    assert wheel_tags.check(grid()) == []


def test_the_grid_is_three_wheels_on_seven_platforms() -> None:
    assert len(grid()) == 3 * 7 + 1


def test_a_missing_wheel_is_named() -> None:
    built = [name for name in grid() if "cp314-cp314t-win_arm64" not in name]
    assert wheel_tags.check(built) == [
        "no cp314-cp314t wheel for win_arm64, which is free-threaded 3.14, which has no stable ABI"
    ]


def test_a_missing_abi_is_named_once_per_platform() -> None:
    built = [name for name in grid() if "cp315-abi3t" not in name]
    complaints = wheel_tags.check(built)
    assert len(complaints) == 7
    assert all("no cp315-abi3t wheel" in complaint for complaint in complaints)


def test_a_version_specific_wheel_nobody_asked_for_is_refused() -> None:
    # What a build that found the wrong interpreter produces, and the
    # reason this check exists: `--no-default-features` against a
    # 3.13 that happened to be on the path builds a wheel that works
    # and claims nothing about any other version.
    built = [*grid(), f"zudb-{VERSION}-cp313-cp313-manylinux_2_28_x86_64.whl"]
    complaints = wheel_tags.check(built)
    assert len(complaints) == 1
    assert complaints[0].startswith(
        f"zudb-{VERSION}-cp313-cp313-manylinux_2_28_x86_64.whl is a cp313-cp313 wheel"
    )


def test_a_gil_enabled_wheel_for_a_single_version_is_refused_too() -> None:
    # The abi3 row falling back to a version-specific build is the same
    # accident and reads the same way in the name.
    built = [*grid(), f"zudb-{VERSION}-cp311-cp311-win_amd64.whl"]
    assert len(wheel_tags.check(built)) == 1


def test_a_platform_this_release_does_not_build_for_is_refused() -> None:
    built = [*grid(), f"zudb-{VERSION}-cp311-abi3-manylinux_2_17_i686.whl"]
    complaints = wheel_tags.check(built)
    assert complaints == [
        f"zudb-{VERSION}-cp311-abi3-manylinux_2_17_i686.whl is for manylinux_2_17_i686, "
        "which is not a platform this release builds for"
    ]


def test_the_macos_deployment_target_is_not_this_check_s_business() -> None:
    # Whichever one the toolchain picked, as long as the wheel is
    # universal2 and there is one of it.
    for target in ("macosx_10_12_universal2", "macosx_11_0_universal2", "macosx_15_0_universal2"):
        built = [name.replace("macosx_10_12_universal2", target) for name in grid()]
        assert wheel_tags.check(built) == []


def test_a_mac_wheel_for_one_architecture_is_not_the_universal_one() -> None:
    # The whole point of the row: two half wheels install on half the
    # machines each, and pip picks one of them without saying so.
    built = [
        name.replace(
            "macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2", "macosx_11_0_arm64"
        )
        for name in grid()
    ]
    complaints = wheel_tags.check(built)
    assert len(complaints) == 6
    assert complaints[0].endswith("which is not a platform this release builds for")


def test_two_wheels_for_one_cell_are_refused() -> None:
    # A universal2 wheel beside the two halves it was fused from is a
    # release that ships the same code twice and installs whichever pip
    # picked.
    built = [*grid(), f"zudb-{VERSION}-cp311-abi3-macosx_14_0_universal2.whl"]
    complaints = wheel_tags.check(built)
    assert len(complaints) == 1
    assert complaints[0].startswith("2 cp311-abi3 wheels for macosx_universal2")


def test_a_release_with_no_sdist_is_refused() -> None:
    built = [name for name in grid() if not name.endswith(".tar.gz")]
    assert wheel_tags.check(built) == [
        "no sdist, and a platform with no wheel of its own builds from one"
    ]


def test_something_that_is_not_ours_is_refused() -> None:
    built = [*grid(), "zu-0.0.1-cp311-abi3-win_amd64.whl"]
    assert wheel_tags.check(built) == [
        "zu-0.0.1-cp311-abi3-win_amd64.whl is neither a wheel of this project nor its sdist"
    ]


def test_one_build_is_held_to_the_cell_it_was_asked_for() -> None:
    built = [f"zudb-{VERSION}-cp314-cp314t-macosx_11_0_universal2.whl"]
    assert wheel_tags.check_one(built, "cp314-cp314t", "macosx_universal2") == []


def test_a_build_that_found_the_wrong_interpreter_is_caught_where_it_happened() -> None:
    # What `--no-default-features` does when the free-threaded 3.14 it
    # asked for is not there and a 3.13 is.
    built = [f"zudb-{VERSION}-cp313-cp313-manylinux_2_28_x86_64.whl"]
    assert wheel_tags.check_one(built, "cp314-cp314t", "manylinux_2_28_x86_64") == [
        f"zudb-{VERSION}-cp313-cp313-manylinux_2_28_x86_64.whl is a cp313-cp313 wheel "
        "and this build asked for cp314-cp314t"
    ]


def test_a_build_for_another_platform_is_caught_too() -> None:
    built = [f"zudb-{VERSION}-cp311-abi3-macosx_11_0_arm64.whl"]
    assert wheel_tags.check_one(built, "cp311-abi3", "macosx_universal2") == [
        f"zudb-{VERSION}-cp311-abi3-macosx_11_0_arm64.whl is for macosx_11_0_arm64 "
        "and this build asked for macosx_universal2"
    ]


def test_a_build_that_produced_nothing_is_caught() -> None:
    assert wheel_tags.check_one([], "cp311-abi3", "win_amd64") == [
        "no cp311-abi3 wheel for win_amd64 was built at all"
    ]


def test_a_build_that_produced_two_wheels_is_caught() -> None:
    # A `dist` that was not emptied between builds, which is two
    # answers to which wheel this row produced and no way to say which
    # of them is the one that gets uploaded.
    built = [
        f"zudb-{VERSION}-cp314-cp314t-manylinux_2_28_x86_64.whl",
        "zudb-0.0.2-cp314-cp314t-manylinux_2_28_x86_64.whl",
    ]
    assert wheel_tags.check_one(built, "cp314-cp314t", "manylinux_2_28_x86_64") == [
        "2 wheels out of one build: zudb-0.0.1-cp314-cp314t-manylinux_2_28_x86_64.whl, "
        "zudb-0.0.2-cp314-cp314t-manylinux_2_28_x86_64.whl"
    ]


def test_the_readme_and_the_check_agree_on_the_three_tags() -> None:
    # The table in the README is what a reader believes, so it is worth
    # one assertion that it says what the gate enforces.
    from pathlib import Path

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    for abi in wheel_tags.ABIS:
        assert f"`{abi}`" in readme
