# The shared corpus, run through this client

The corpus is one set of hand written YAML files in the engine's repository, under `conformance/cases`. Every client reads the same files and runs the same statements, so a value that survives one binding and not another is a diff rather than an argument. This directory is the Python end of that: a reader for the subset of YAML the cases are written in, a decoder for the value encoding, and a runner that reports what happened in the form the reference runner reports it.

It is a development tool and not part of the wheel. `pyproject.toml` packages `python/` and nothing else, so `conformance` is on the path when the repository is checked out and absent when `zudb` is installed. Nothing under `zudb` imports it.

## Running it

The cases live in the engine's repository, pinned in `Cargo.toml` to the same revision this client builds against:

```
git clone https://github.com/tamnd/zu /tmp/zu
git -C /tmp/zu checkout 6ceb6d50abe20cfbef97c3d0d033d051c0649521
python -m conformance /tmp/zu/conformance/cases
```

which prints every case that did not pass and then one line saying what the run came to:

```
945 cases, 940 passed, 0 failed, 5 unsupported
```

`--strict` turns an unsupported case into a failed run, `--quiet` prints the summary alone, and `--work DIR` keeps the databases the cases were run against instead of removing them, which is what to reach for when a failure wants opening.

The exit code is 0 when nothing failed, and 1 when something did or when the corpus will not read. One and not two for a corpus that will not read, because the reference runner exits one and a report compared line for line is worth less if the two runners disagree about what the run came to.

## What it is checking

Three things, and the third is the one that matters.

A client can decode a value and pass every case, because the case and the answer both went through the same decoder. So the reader here is a third implementation of the corpus format rather than a consumer of one: it refuses what `crates/zu-corpus/src/yaml.rs` refuses, with the same words and the same line numbers, and the tests in `tests/test_conformance.py` are that file's own tables ported case for case. A reader that grew a hole would pass its own tests and fail those.

The value encoding is the same again. An INT64 written bare is refused, a value wider than the type it claims is refused, a float is exact or it is not a float, and a temporal is written the way the engine prints it. What a report prints is the encoding's own spelling, so a failure can be pasted back into a case.

And the report itself is compared. Two mutation sweeps of the corpus, one that corrupts every third payload so the readers refuse and one that rewrites every row value so the reports compare, produce output this runner and the Rust one agree on line for line, with the single exception below. Five defects in this client were found that way and none of them by its own tests: a helper shadowed by a loop variable so two refusals would have raised, a float printed `1e+16` where the engine prints `1e16`, a time printed with six fractional digits where the engine prints nine, a zero offset printed `+00:00` where the engine prints `Z`, and a duration printed with the fields that are zero left in.

## What this client cannot answer

Five cases, all of them a time written finer than a microsecond:

- `temporal/local-time-nanoseconds`
- `temporal/a-time-carries-a-single-nanosecond`
- `param/localtime-to-the-nanosecond`
- `stored/a-localtime-column-keeps-every-digit`
- `stored/the-columns-of-one-row-belong-to-that-row`

The engine keeps a time to the nanosecond and Python's `datetime` keeps one to the microsecond, so `LOCAL TIME '12:34:56.123456789'` comes back as `datetime.time(12, 34, 56, 123456)`. That is the value mapping this client documents rather than a defect in it, and it is what `temporal/local-time-nanoseconds` says out loud it was written to catch.

The trap is that a runner catches it only if it tries to. Decoding the case's own expectation into a `datetime` truncates it too, and the case then passes by comparing one truncated value against another. So a temporal payload with a digit past the sixth decodes to a value that equals nothing, including itself, and a case holding one is reported unsupported with the value named. Unsupported and not failed, because nothing went wrong: the engine answered, and Python's `datetime` is where the digits went.

A load is the one place a value this client cannot hold still goes in, truncated. A column has to be in the file for the suite to have a graph at all, and refusing it would take out the thirty six cases of `stored` rather than the two that read that column back. Those two say so on their own, because their own expectation is a value nothing equals.

## The one check that is weaker here than in the engine

A case naming a condition writes its GQLSTATUS, and the reference reader checks that code against the table the standard defines, which lives in `zu-common`. This reader checks the shape, five characters of digits and capitals, because the table is not something the client has. A code of the right shape that no standard defines is caught by the reference runner and not by this one. The C reader in `conformance/c` takes the same position.

## The files

`reader.py` is the YAML subset: block mappings, block sequences, scalars, and a refusal with a line number for everything else. It does not use PyYAML, which would read these files and a good deal more besides, and would hand back `9223372036854775807` as a float on the way.

`values.py` is the `{type, value}` encoding, both directions, and the comparison. The comparison is not `==`: `NaN` matches `NaN`, `-0.0` does not match `0.0`, and a `bool` does not match an `int`, which is a rule Python needs and the reference runner does not.

`cases.py` is what a case is, and `runner.py` runs them, one database per case with a fresh copy of the suite's load, so a case that leaked a table into the next one would be a failure that moves when the file is reordered.
