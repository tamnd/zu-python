"""What a release is allowed to have built.

Three wheels per platform, which is more than it sounds like it should
be and is not optional: the free-threaded build has no stable ABI until
CPython 3.15 and PEP 803's `abi3t`, so 3.14t needs a version-specific
wheel of its own. From 3.15 one wheel serves both builds and says so in
two ABI tags at once, `cp315-abi3.abi3t`, which is the whole point of
PEP 803 and the reason this stops at three rather than growing a fourth.
Three, and no more than three. A version-specific
wheel nobody asked for is what happens when a build finds the wrong
interpreter and quietly falls back, and the failure is silent in every
other way: the wheel installs, it works on the machine that built it,
and it is wrong for every version it claims nothing about.

So the release job builds the grid and this checks it, by name, both
ways: every cell filled, and nothing outside the grid.

    python tools/wheel_tags.py dist                             the grid
    python tools/wheel_tags.py dist cp311-abi3 win_amd64        one cell
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: The ABIs, as `{python tag}-{abi tag}`, which is what the middle of a
#: wheel's name says and the only part of it that says which
#: interpreters the wheel is for.
ABIS = {
    "cp311-abi3": "the stable ABI, CPython 3.11 through 3.14",
    "cp314-cp314t": "free-threaded 3.14, which has no stable ABI",
    "cp315-abi3.abi3t": "the stable ABI from 3.15, GIL-enabled and free-threaded at once",
}

#: The platforms, as patterns rather than strings, because the macOS
#: tag carries the deployment target the toolchain picked and that is
#: not this file's business.
PLATFORMS = {
    "manylinux_2_28_x86_64": r"manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64": r"manylinux_2_28_aarch64",
    "musllinux_1_2_x86_64": r"musllinux_1_2_x86_64",
    "musllinux_1_2_aarch64": r"musllinux_1_2_aarch64",
    "macosx_universal2": r"macosx_\d+_\d+_universal2",
    "win_amd64": r"win_amd64",
    "win_arm64": r"win_arm64",
}

#: The source distribution, which is the wheel for everything else: a
#: platform with no row above builds from it with a Rust toolchain and
#: nothing else.
SDIST = re.compile(r"^zudb-[^-]+\.tar\.gz$")

WHEEL = re.compile(r"^zudb-[^-]+-([^-]+-[^-]+)-(.+)\.whl$")


def members(tag: str) -> tuple[str, frozenset[str]]:
    """A `{python tag}-{abi tag}` as what it means: the interpreter it
    starts from, and the set of ABIs it claims."""
    python, _, abis = tag.partition("-")
    return python, frozenset(abis.split("."))


def abi_of(tag: str) -> str | None:
    """Which ABI a tag is, or `None` for one this release does not build.

    An ABI tag is a set, dots between its members, the same way a
    platform tag is, and PEP 803's wheel is where that stops being a
    detail: it arrives as `cp315-abi3.abi3t`, which is one wheel saying
    it is the stable ABI for the GIL-enabled build and for the
    free-threaded build at once. Which member maturin writes first is
    not something to hold a release to, so the set is what is compared,
    and a wheel that claims only one of the two is not this one.
    """
    for name in ABIS:
        if members(tag) == members(name):
            return name
    return None


def platform_of(tag: str) -> str | None:
    """Which platform a tag is, or `None` for one that is no platform
    this release builds for.

    A platform tag is a set rather than a tag, dots between its
    members, and a universal2 wheel is where that stops being a detail:
    it arrives as
    `macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2`,
    which is one wheel saying the three true things about itself. A set
    is the platform whose pattern one of its members matches, and a
    wheel for one architecture matches none of them.
    """
    for name, pattern in PLATFORMS.items():
        if any(re.fullmatch(pattern, one) for one in tag.split(".")):
            return name
    return None


def check(names: list[str]) -> list[str]:
    """Every complaint about what was built, in the order they are worth
    reading. An empty list is a release."""
    complaints = []
    built: dict[tuple[str, str], list[str]] = {}
    sdists = [name for name in names if SDIST.fullmatch(name)]

    for name in sorted(names):
        if name in sdists:
            continue
        found = WHEEL.fullmatch(name)
        if not found:
            complaints.append(f"{name} is neither a wheel of this project nor its sdist")
            continue
        abi, platform = found.group(1), found.group(2)
        which = abi_of(abi)
        if which is None:
            wanted = ", ".join(ABIS)
            complaints.append(
                f"{name} is a {abi} wheel, and this release builds {wanted} and nothing else"
            )
            continue
        where = platform_of(platform)
        if where is None:
            complaints.append(
                f"{name} is for {platform}, which is not a platform this release builds for"
            )
            continue
        built.setdefault((which, where), []).append(name)

    for abi, what in ABIS.items():
        for platform in PLATFORMS:
            made = built.get((abi, platform), [])
            if not made:
                complaints.append(f"no {abi} wheel for {platform}, which is {what}")
            elif len(made) > 1:
                complaints.append(f"{len(made)} {abi} wheels for {platform}: {', '.join(made)}")

    if not sdists:
        complaints.append("no sdist, and a platform with no wheel of its own builds from one")
    elif len(sdists) > 1:
        complaints.append(f"{len(sdists)} sdists: {', '.join(sorted(sdists))}")

    return complaints


def check_one(names: list[str], abi: str, platform: str) -> list[str]:
    """Every complaint about one cell of the grid, which is what a
    single build produced and all it can be held to.

    Worth checking there and not only at the end, because a build that
    fell back to another interpreter says so in the name of the wheel
    and nowhere else, and the row that did it is the one that knows
    which interpreter it asked for.
    """
    if abi not in ABIS:
        return [f"{abi} is not one of the ABIs this release builds: {', '.join(ABIS)}"]
    if platform not in PLATFORMS:
        return [f"{platform} is not one of the platforms: {', '.join(PLATFORMS)}"]
    complaints = []
    for name in sorted(names):
        found = WHEEL.fullmatch(name)
        if not found:
            complaints.append(f"{name} is not a wheel of this project")
        elif abi_of(found.group(1)) != abi:
            complaints.append(f"{name} is a {found.group(1)} wheel and this build asked for {abi}")
        elif platform_of(found.group(2)) != platform:
            complaints.append(f"{name} is for {found.group(2)} and this build asked for {platform}")
    if not names:
        complaints.append(f"no {abi} wheel for {platform} was built at all")
    elif len(names) > 1 and not complaints:
        complaints.append(f"{len(names)} wheels out of one build: {', '.join(sorted(names))}")
    return complaints


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 4):
        print(f"usage: {argv[0]} <directory> [<abi tag> <platform>]", file=sys.stderr)
        return 2
    into = Path(argv[1])
    names = sorted(path.name for path in into.rglob("*") if path.is_file())
    if len(argv) == 4:
        complaints = check_one(names, argv[2], argv[3])
        expected = 1
    else:
        if not names:
            print(f"{into} holds nothing, so nothing was built", file=sys.stderr)
            return 1
        complaints = check(names)
        expected = len(ABIS) * len(PLATFORMS) + 1
    for complaint in complaints:
        print(complaint, file=sys.stderr)
    print(f"{len(names)} built, {expected} expected, {len(complaints)} complaints")
    return 1 if complaints else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
