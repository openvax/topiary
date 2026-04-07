#!/bin/bash
set -o errexit

ruff check topiary/ tests/

echo 'Passes ruff check'
