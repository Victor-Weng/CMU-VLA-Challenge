#!/usr/bin/env bash
set -e

# Source ROS Noetic environment
if [ -f "/opt/ros/noetic/setup.bash" ]; then
  source /opt/ros/noetic/setup.bash
fi

# Move to working directory set by docker-compose (defaults to repo root)
cd "$(pwd)"

# If a catkin workspace exists at ai_module, build it once if not already built
if [ -d "ai_module/src" ]; then
  pushd ai_module >/dev/null
  if [ ! -f "devel/setup.bash" ] || [ ! -d "build" ]; then
    echo "[entrypoint] Building catkin workspace in $(pwd)"
    CORES=$(nproc || echo 2)
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
    if command -v ccache >/dev/null 2>&1; then
      echo "[entrypoint] Using ccache to speed up C++ builds"
      CMAKE_ARGS+=" -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    fi
    catkin_make -j"$CORES" -l"$CORES" $CMAKE_ARGS
  fi
  # Source the workspace overlay
  if [ -f "devel/setup.bash" ]; then
    source devel/setup.bash
  fi
  popd >/dev/null
fi

# Execute the provided command
exec "$@"
