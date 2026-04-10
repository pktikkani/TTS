#!/bin/bash
# Run this on a new machine after cloning the repo to set up Claude Code memory
# Usage: bash .claude/setup-memory.sh

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
# Claude Code uses the project path with / replaced by - and leading -
MEMORY_PATH="$HOME/.claude/projects/-$(echo "$REPO_DIR" | sed 's|^/||; s|/|-|g')/memory"

echo "Setting up Claude Code memory..."
echo "Repo dir: $REPO_DIR"
echo "Memory path: $MEMORY_PATH"

mkdir -p "$MEMORY_PATH"
cp "$REPO_DIR/.claude/memory/"* "$MEMORY_PATH/"

echo "Done! Memory files copied to $MEMORY_PATH"
ls -la "$MEMORY_PATH"
