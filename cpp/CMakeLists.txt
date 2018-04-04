cmake_minimum_required(VERSION 3.9)
project(OCVSRC)

set(CMAKE_CXX_STANDARD 11)
add_executable(OCVSRC main.cpp)

set(Anaconda /Users/mac/.pyenv/versions/anaconda2-5.0.1)
SET(OpenCV_DIR ${Anaconda}/share/OpenCV/)
find_package(OpenCV 3.3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(OCVSRC ${OpenCV_LIBS})

#set(BOOST_ROOT ${Anaconda})
#set(BOOST_INCLUDEDIR ${Anaconda}/include)
#set(BOOST_LIBRARYDIR ${Anaconda}/lib)
#set(Boost_NO_SYSTEM_PATHS ON)
#find_package(Boost REQUIRED COMPONENTS regex filesystem system)
#include_directories(${Boost_INCLUDE_DIRS})
#target_link_libraries(OCVSRC ${Boost_LIBRARIES})
