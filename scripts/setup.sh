#!/usr/bin/env bash
set -e

# Installs `hatch` globally using `pipx`

if ! command -v hatch &>/dev/null; then
    if ! command -v pipx &>/dev/null; then
        echo "Installing 'pipx'..."
        python3 -m pip install --user pipx
    fi
    echo "Installing 'hatch'..."
    pipx install hatch
else
    echo "'hatch' is already installed: $(which hatch)"
fi

