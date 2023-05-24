cmake_minimum_required(VERSION 3.5)
project(vl53l0x_driver)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

message("COMPILING SHARED LIBRARY WITH COMMAND `make`")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Makefile
  ${CMAKE_CURRENT_BINARY_DIR}/Makefile
)

execute_process(
  COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set(output_bin_dir ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(output_obj_dir ${CMAKE_CURRENT_SOURCE_DIR}/obj)

install(
  DIRECTORY ${output_bin_dir}
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY ${output_obj_dir}
  DESTINATION share/${PROJECT_NAME} 
)

ament_python_install_package(${PROJECT_NAME})

ament_package()
