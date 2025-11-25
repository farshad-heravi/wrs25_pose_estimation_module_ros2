cmake_minimum_required(VERSION 3.8)
project(wrs25_pose_estimation_module_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Install launch files so they are available as part of the package
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(PYTHON_INSTALL_DIR lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages)

# Generate action interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/BoxVision.action"
  "action/SprayVision.action"
  "action/VisionCheck.action"
  DEPENDENCIES action_msgs geometry_msgs
)

# Install action files
install(DIRECTORY action
  DESTINATION share/${PROJECT_NAME}/
)

# Install Python modules
install(DIRECTORY ${PROJECT_NAME}/
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/
)

# Install Python executables
install(PROGRAMS
  ${PROJECT_NAME}/box_action_server.py
  ${PROJECT_NAME}/spray_action_server.py
  ${PROJECT_NAME}/vision_check_server.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

