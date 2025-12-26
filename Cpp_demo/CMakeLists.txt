cmake_minimum_required(VERSION 3.10)

project(casadi LANGUAGES CXX)

# ============================================================
# 基础配置
# ============================================================

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

# ============================================================
# 第三方路径（来自 VS settings）
# ============================================================

set(vsImguiRoot "" CACHE PATH "ImGui root directory")

set(vsGlfwRoot "" CACHE PATH "GLFW root directory")

set(vsImplotRoot "" CACHE PATH "ImPlot root directory")

# 如有变化，只改这一个路径即可
set(vsCasadiRoot "" CACHE PATH "Path to CasADi installation root")
set(vsCasadi_INCLUDE_DIR "${vsCasadiRoot}/include")
set(vsCasadi_LIB_DIR     "${vsCasadiRoot}/lib")
file(GLOB vsCasadi_LIBS "${vsCasadi_LIB_DIR}/Release/*.lib")

# ============================================================
# GLFW
# ============================================================

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

add_subdirectory(${vsGlfwRoot} ${CMAKE_CURRENT_BINARY_DIR}/glfw)

# ============================================================
# ImGui（明确列出源码）
# ============================================================

set(IMGUI_SOURCES
    ${vsImguiRoot}/imgui.cpp
    ${vsImguiRoot}/imgui_draw.cpp
    ${vsImguiRoot}/imgui_tables.cpp
    ${vsImguiRoot}/imgui_widgets.cpp
    ${vsImguiRoot}/imgui_demo.cpp
    ${vsImguiRoot}/backends/imgui_impl_glfw.cpp
    ${vsImguiRoot}/backends/imgui_impl_opengl3.cpp
)

# ============================================================
# ImPlot（明确列出源码）
# ============================================================

set(IMPLOT_SOURCES
    ${vsImplotRoot}/implot.cpp
    ${vsImplotRoot}/implot_items.cpp
    ${vsImplotRoot}/implot_demo.cpp
)

# ============================================================
# 输出目录
# ============================================================
# 让 VS 把 exe 放在 <工程根>/Debug 和 <工程根>/Release，配合你的 launch.json
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# 项目源文件（使用通配符）
# ============================================================

file(GLOB_RECURSE PROJECT_CPP_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE PROJECT_HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
)

# ============================================================
# 目标定义
# ============================================================
add_executable(${PROJECT_NAME}
    ${PROJECT_CPP_FILES}
    ${PROJECT_HEADER_FILES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${vsImguiRoot}
        ${vsImguiRoot}/backends
        ${vsImplotRoot}
        ${vsGlfwRoot}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${vsCasadi_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        glfw
        opengl32
        ${vsCasadi_LIBS}
)
