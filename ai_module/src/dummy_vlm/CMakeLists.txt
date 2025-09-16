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
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Use C++11 for lambda and modern constructs
add_compile_options(-std=c++11)

add_executable(dummyVLM src/dummyVLM.cpp)
add_executable(frontier_explorer src/frontier_explorer.cpp)
add_executable(pc_to_grid src/pc_to_grid.cpp)
target_link_libraries(dummyVLM
  ${catkin_LIBRARIES}
)
target_link_libraries(frontier_explorer ${catkin_LIBRARIES})
add_dependencies(frontier_explorer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(pc_to_grid ${catkin_LIBRARIES})
add_dependencies(pc_to_grid ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
