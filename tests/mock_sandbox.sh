#! /usr/bin/env bash

script_dir=$(dirname "$0")
proposal="${1}"
out_file="${script_dir}/${proposal}"

# Sleep for a bit to ensure that jobs can run concurrently
sleep 0.5

# Atomically append the arguments it was called with to a file. Each line in the
# file thus corresponds to a DAMNIT job.
(
    flock -x 200
    echo "${@}" >> ${out_file}
) 200>/tmp/damnit_tests_lockfile
