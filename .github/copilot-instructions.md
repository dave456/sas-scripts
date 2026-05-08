# GitHub Copilot Instructions

## Project Overview

This repository contains scripts for [Seti Astro Suite Pro](https://www.setiastro.com/seti-astro-suite-pro), an open-source astrophotography image processing application. These python scripts (`.py` files) are GUI scripts driven by the `setiastro.saspro` Python library.

## Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of picking silently.
- If a simpler solution exists, say so.
- If requirements are unclear, stop and ask a clarifying question.
- When porting code, attempt to keep the new code as close to the source implementation as practical to reduce future merge conflicts. If a change is needed, explain why.
- Deduplicate obvious repetition before finalizing:
    - If two or more callbacks/functions differ only by target widget, field name, or constant argument, merge into one parameterized helper.
    - Prefer passing context (attribute name, widget reference, enum/value) instead of creating one-off wrapper handlers.
- Mandatory self-review pass before returning code:
    - Scan for near-identical methods created during incremental implementation.
    - Collapse duplicates unless separation is required for clarity, threading, or future behavior divergence.
    - If duplicates are intentionally kept, state the reason explicitly.

## Simplicity First

**Write the minimum code that solves the requested problem. Nothing speculative.**

- Do not add code beyond the request.
- Avoid abstractions for one-time use.
- When porting code, keep the new code as close to the source implementation as practical to reduce future merge conflicts.

## Repository Structure

| File | Description |
|------|-------------|
| ContinuumSubtraction.py | Linear continuum subtraction tool |
| Remove_Banding.py | Removing linear banding noise |
| SCUNet_Denoise_SAS.py | Port of siril SCUNet denoiser |
| Star_Reducer.py | Star reduction utility |
| template.py | Template for building modal dialogs for SASpro |
| Veralux_HyperMetric_Stretch_SAS.py | Port of Veralux HyperMetric stretch |

## Python Script Conventions

### Imports and Dependencies

- Standard third-party libraries: `astropy`, `numpy`, `PyQt6` (for Qt-based GUIs)
- The python API documentation for SASpro can be found here: 

### Image Data

- Use `self.ctx.get_image()` to get the current image as a numpy array
- Use `self.ctx.set_image` to set the image data back in SASpro after processing
- Use `self.ctx.log() to log messages to the SASpro console

### GUI Style

Prefer PyQt6 for all new scripts:

1. **PyQt6** (NarrowBandMixer, Star_Reducer, etc.):
   - Subclass `QDialog` for the main window
   - Use `QMessageBox` for error/info dialogs

### Threading

- Long-running operations (image processing, external CLI tools) must run in a background thread to keep the GUI responsive
- Pattern used: `threading.Thread(target=lambda: asyncio.run(self.ApplyChanges()), daemon=True).start()`
- Disable UI controls before starting background work and re-enable in the `finally` block

### External Tools Integration

When calling external CLI tools (GraXpert, Cosmic Clarity):
- Save the current Siril image to a temp FITS file using `astropy.io.fits`
- Call the external tool via `subprocess.run()` or `asyncio.create_subprocess_exec()`
- Always clean up temp files in a `finally` block

### Script Entry Point

All Python scripts follow this pattern:

```python
def run(ctx):
    """
    SASpro entry point.
    """
    w = ScriptWindow(ctx, parent=ctx.app)
    w.setModal(False)
    w.setWindowModality(Qt.WindowModality.NonModal)
    w.show()

    # Keep a reference on the context so Python doesn't GC the window
    try:
        setattr(ctx, "_star_reducer_window", w)
    except Exception:
        pass

    return w
```

## License

All scripts are licensed under GPL-3.0.
