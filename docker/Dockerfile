FROM ros:noetic-ros-core-focal

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-noetic-ros-base=1.5.0-1* \
  # Core middleware we need at runtime
  ros-noetic-tf \
  ros-noetic-tf2-ros \
  ros-noetic-cv-bridge \
  ros-noetic-image-transport \
  ros-noetic-pcl-ros \
  ros-noetic-pcl-conversions \
  ros-noetic-visualization-msgs \
  ros-noetic-nav-msgs \
  ros-noetic-geometry-msgs \
  ros-noetic-std-srvs \
  # System libs for OpenCV and GUI-less image ops
  python3-numpy \
  libgl1 \
  libglib2.0-0 \
  # Networking for Mistral client
  libcurl4 \
  # C++ JSON library header-only package for <nlohmann/json.hpp>
  nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: pip if we need to install Python packages from requirements later
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*

# Install Python deps: headless OpenCV and Ultralytics (YOLOv8)
RUN pip3 install --no-cache-dir opencv-python-headless ultralytics

# Source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Copy entrypoint to auto-build catkin workspace and source overlays
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
