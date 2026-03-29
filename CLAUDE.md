# CLAUDE.md

This file provides context for Claude Code (Anthropic's AI coding assistant) when working in this repository.

## Project Overview

This repository contains scripts for [Seti Astro Suite Pro](https://www.setiastro.com/seti-astro-suite-pro), an open-source astrophotography image processing application. These python scripts (`.py` files) are GUI scripts driven by the `setiastro.saspro` Python library.

## Repository Structure

| File | Description |
|------|-------------|
| `Remove_Banding.py` | Removing linear banding noise |
| `Star_Reducer.py` | Star reduction utility |
| `ContinuumSubtraction` | Linear continuum subtraction tool |
| `template.py` | Template for building modal dialogs for SASpro |

## Development Notes

- Python scripts use `setiastro.saspro` for Seti Astro Suite Pro communication and `PyQt6` for GUIs.
- The `dev/` directory is excluded from version control (see `.gitignore`).
