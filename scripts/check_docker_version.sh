#!/usr/bin/env bash

DOCKER_MINIMUM_REQUIRED_VERSION=19.03

# Copied from https://stackoverflow.com/a/24067243
function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

current_docker_version=`docker version --format '{{.Server.Version}}'`
if version_gt $DOCKER_MINIMUM_REQUIRED_VERSION $current_docker_version; then
     echo "Docker version >=${DOCKER_MINIMUM_REQUIRED_VERSION} required for for the --gpus option."
fi
