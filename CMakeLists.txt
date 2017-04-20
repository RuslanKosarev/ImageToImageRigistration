cmake_minimum_required(VERSION 2.8)

project(Registration)

# Find AGTK
find_package(AGTK REQUIRED)
include(${AGTK_USE_FILE})

add_executable(imageRegistration imageRegistration.cxx)
target_link_libraries(imageRegistration ${AGTK_LIBRARIES})

add_executable(registrationv4 registrationv4.cxx)
target_link_libraries(registrationv4 ${AGTK_LIBRARIES})

add_executable(MRIRegistration-translation MRIRegistration-translation.cxx)
target_link_libraries(MRIRegistration-translation ${AGTK_LIBRARIES})

add_executable(processing processing.cxx)
target_link_libraries(processing ${AGTK_LIBRARIES})
