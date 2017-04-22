cmake_minimum_required(VERSION 2.8)

project(Registration)

# Find AGTK
find_package(AGTK REQUIRED)
include(${AGTK_USE_FILE})

add_executable(MattesMutualInformationImageRegistration MattesMutualInformationImageRegistration.cxx)
target_link_libraries(MattesMutualInformationImageRegistration ${AGTK_LIBRARIES})
