cmake_minimum_required(VERSION 3.0.2)
project(dummy_vlm)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf
  message_filters
)

# System dependency for HTTP calls in dummyVLM.cpp
find_package(CURL REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CURL_INCLUDE_DIRS}
)

# Use C++11 for lambda and modern constructs
add_compile_options(-std=c++11)

add_executable(dummyVLM src/dummyVLM.cpp)
add_executable(frontier_explorer src/frontier_explorer.cpp)
add_executable(pc_to_grid src/pc_to_grid.cpp)
add_executable(image_lidar_capture src/image_lidar_capture.cpp)
target_link_libraries(dummyVLM
  ${catkin_LIBRARIES}
  ${CURL_LIBRARIES}
)
target_link_libraries(frontier_explorer ${catkin_LIBRARIES})
add_dependencies(frontier_explorer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pc_to_grid ${catkin_LIBRARIES})
add_dependencies(pc_to_grid ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(image_lidar_capture ${catkin_LIBRARIES})
add_dependencies(image_lidar_capture ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Install Python scripts (ROS nodes)
catkin_install_python(PROGRAMS
  scripts/camera_save_and_detect.py
  scripts/paired_detect_and_depth.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
