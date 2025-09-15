cmake_minimum_required(VERSION 3.0.2)
project(dummy_vlm)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs nav_msgs geometry_msgs visualization_msgs tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(dummyVLM src/dummyVLM.cpp)

target_link_libraries(dummyVLM
  ${catkin_LIBRARIES}
)
