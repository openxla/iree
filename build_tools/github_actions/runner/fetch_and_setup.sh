#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script is actually part of the VM image and fetches the rest of the
# configuration. It is invoked on startup through a one-line startup script that
# calls it. Longer term, we may want to have an explicit deployment of new
# scripts instead of fetching them directly from HEAD.

set -euo pipefail
shopt -s extglob

SCRIPT="$( readlink -f -- "$0"; )"
SCRIPT_BASENAME="$(basename ${SCRIPT})"

if [[ "$(whoami)" != "runner" ]]; then
  echo "Current user is not 'runner'. Rerunning script as 'runner'."
  sudo su runner --shell /bin/bash --command "${SCRIPT}"
  exit
fi

cd "${HOME}"
rm -rf -v !("actions-runner"|"${SCRIPT_BASENAME}")

cd /tmp/
rm -rf /tmp/iree
git clone https://github.com/iree-org/iree.git

cd iree/build_tools/github_actions/runner/
# Bash doesn't read the whole script into memory when executing, so if you
# overwrite it during execution, madness ensues.
cp -r !("${SCRIPT_BASENAME}") "${HOME}/"

cd "${HOME}"
rm -rf /tmp/iree

./setup.sh
