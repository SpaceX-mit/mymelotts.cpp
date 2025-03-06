
cmake_minimum_required(VERSION 3.10)
project(MeloTTS VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

# 查找ONNX Runtime
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS /usr/local/lib)
if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime library not found. Please install ONNX Runtime.")
endif()

# 设置包含目录
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  /usr/local/include
  /usr/local/include/onnxruntime
)

# 设置库目录
link_directories(
  /usr/local/lib
)

# 源文件
set(SOURCES
  src/melotts.cpp
  src/text_processor.cpp
  src/acoustic_model.cpp
  src/vocoder.cpp
)

# 头文件
set(HEADERS
  include/melotts.h
  include/text_processor.h
  include/OnnxWrapper.hpp
  include/MeloTTSConfig.h
  include/AudioFile.h
  include/Lexicon.hpp
  include/EngineWrapper.hpp
)

# 创建库目标
add_library(melotts SHARED ${SOURCES})
target_link_libraries(melotts ${ONNXRUNTIME_LIBRARY})

# 创建可执行文件目标
add_executable(melotts_cli src/main.cpp)
target_link_libraries(melotts_cli melotts)

# 创建测试程序
add_executable(test_onnx src/test_onnx.cpp)
target_link_libraries(test_onnx ${ONNXRUNTIME_LIBRARY})

# 安装
install(TARGETS melotts melotts_cli
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(FILES ${HEADERS} DESTINATION include/melotts)

# 添加示例目标
add_executable(example src/example.cpp)
target_link_libraries(example melotts)

