cmake_minimum_required(VERSION 3.10)

project(casadi LANGUAGES CXX)

# 生成 compile_commands.json（VS Code IntelliSense 必需）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC 编码与告警处理
if(MSVC)
    add_compile_options(
        "/source-charset:utf-8"  # 源文件按 UTF-8 读取
        "/wd4828"             # 屏蔽 UTF-8 相关告警
    )
endif()

# OpenCV：根据你 launch.json 中的路径，设置 OpenCV_DIR，然后 find_package
# 如有变化，只改这一个路径即可
set(vsCasadiRoot "" CACHE PATH "Path to CasADi installation root")
set(vsCasadi_INCLUDE_DIR "${vsCasadiRoot}/include")
set(vsCasadi_LIB_DIR     "${vsCasadiRoot}/lib")
file(GLOB vsCasadi_LIBS "${vsCasadi_LIB_DIR}/Release/*.lib")

# ================= 输出与包含路径 =================
# 让 VS 把 exe 放在 <工程根>/Debug 和 <工程根>/Release，配合你的 launch.json
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# 头文件路径
set(PROJECT_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${vsCasadi_INCLUDE_DIR}
)

# ================= 源文件、UI、资源 =================
file(GLOB SRC_FILES
    *.cpp
)

file(GLOB HEADER_FILES
    *.h
)

# ================= 目标与链接 =================
add_executable(${PROJECT_NAME}
    ${SRC_FILES}
    ${HEADER_FILES}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${vsCasadi_LIBS}
)
