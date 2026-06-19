# Security Policy

This project is a research-oriented simulator and analysis toolkit. It does not
process untrusted network traffic, but it does parse local config files and may
launch helper Python processes for Parquet conversion.

## Supported Versions

Security fixes target the current `main` branch and active release branches, if
any exist. Older snapshots are not supported unless a maintainer explicitly says
otherwise.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting or the repository Security tab when
available. If that is unavailable, contact the maintainer privately through
GitHub with:

- affected commit or version
- operating system and toolchain
- reproduction steps
- expected impact
- any suggested mitigation

You should receive an initial response within a reasonable maintainer window.
Public disclosure should wait until a fix or mitigation is available.
