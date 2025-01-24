# -----------------------------------------------------------------------------
# @brief  : Root cmake file.
# @author : Enrico Fraccaroli
# -----------------------------------------------------------------------------

# Set the minimum CMake version, the project name and default build type.
cmake_minimum_required(VERSION 3.1...3.18)

# Set the project name.
project(fsmlib CXX)

# Set the default build type to Debug.
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

# -----------------------------------------------------------------------------
# ENABLE FETCH CONTENT
# -----------------------------------------------------------------------------

# We need this in order to import external projects.
include(FetchContent)
# Hide fetchcontent variables.
mark_as_advanced(FORCE
    FETCHCONTENT_QUIET
    FETCHCONTENT_BASE_DIR
    FETCHCONTENT_FULLY_DISCONNECTED
    FETCHCONTENT_UPDATES_DISCONNECTED
)

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(STRICT_WARNINGS "Enable strict compiler warnings" ON)
option(WARNINGS_AS_ERRORS "Treat all warnings as errors" OFF)

# -----------------------------------------------------------------------------
# DEPENDENCY (SYSTEM LIBRARIES)
# -----------------------------------------------------------------------------

find_package(Doxygen)

# -----------------------------------------------------------------------------
# LIBRARY
# -----------------------------------------------------------------------------

# Add the C++ Library.
add_library(fsmlib INTERFACE)
add_library(fsmlib::fsmlib ALIAS fsmlib)
# Inlcude header directories.
target_include_directories(fsmlib INTERFACE ${PROJECT_SOURCE_DIR}/include)
# Set the library to use c++-20
target_compile_features(fsmlib INTERFACE cxx_std_17)

# -----------------------------------------------------------------------------
# COMPILATION FLAGS
# -----------------------------------------------------------------------------

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Disable warnings for MSVC-specific "safe" functions like strcpy_s, etc.,
    # which are not portable and may clutter warning logs.
    target_compile_definitions(fsmlib INTERFACE _CRT_SECURE_NO_WARNINGS)

    if(WARNINGS_AS_ERRORS)
        # Treat all warnings as errors to enforce stricter code quality.
        target_compile_options(fsmlib INTERFACE /WX)
    endif()

    if(STRICT_WARNINGS)
        # Enable external header management to suppress warnings in system and
        # external headers, making it easier to focus on project-specific issues.
        target_compile_options(fsmlib INTERFACE /experimental:external)
        target_compile_options(fsmlib INTERFACE /external:I ${CMAKE_BINARY_DIR})
        target_compile_options(fsmlib INTERFACE /external:anglebrackets)
        target_compile_options(fsmlib INTERFACE /external:W0)

        # Use a high warning level to catch as many potential issues as possible.
        target_compile_options(fsmlib INTERFACE /W4)

        # Enforce standards-compliant behavior to avoid relying on MSVC-specific extensions.
        target_compile_options(fsmlib INTERFACE /permissive-)
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(WARNINGS_AS_ERRORS)
        # Treat all warnings as errors to enforce stricter code quality.
        target_compile_options(fsmlib INTERFACE -Werror)
    endif()

    if(STRICT_WARNINGS)
        # Enable a broad set of warnings to catch common and subtle issues:
        target_compile_options(fsmlib INTERFACE
            -Wall                # Enable most general-purpose warnings.
            -Wextra              # Enable extra warnings not included in -Wall.
            -Wconversion         # Warn about implicit type conversions that may lose data.
            -pedantic            # Enforce strict compliance with the C++ standard.
            -Wshadow             # Warn about variable shadowing, which can cause subtle bugs.
            -Wnon-virtual-dtor   # Warn when a class with virtual functions lacks a virtual destructor.
            -Wnull-dereference   # Warn about potential null pointer dereferences.
            -Wformat=2           # Enable strict checks for printf/scanf format strings.
            -Woverloaded-virtual # Warn when a derived class function hides a base class virtual function.
            -Wfloat-equal        # Warn about direct comparisons of floating-point values, which can be imprecise.
        )

    endif()
endif()


# -----------------------------------------------------------------------------
# EXAMPLES
# -----------------------------------------------------------------------------

if(BUILD_EXAMPLES)
    # Add the example.
    add_executable(fsmlib_usage examples/usage/main.cpp)
    target_include_directories(fsmlib_usage PUBLIC ${PROJECT_SOURCE_DIR}/examples)
    target_link_libraries(fsmlib_usage PUBLIC fsmlib)
endif()

# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

if(BUILD_TESTS)

    # CMake has support for adding tests to a project.
    enable_testing()

    # Add the test.
    add_executable(fsmlib_test_fsmlib ${PROJECT_SOURCE_DIR}/tests/test_fsmlib.cpp)
    target_include_directories(fsmlib_test_fsmlib PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(fsmlib_test_fsmlib fsmlib)
    add_test(fsmlib_test_fsmlib_run fsmlib_test_fsmlib)

    # Add the test.
    add_executable(fsmlib_test_math ${PROJECT_SOURCE_DIR}/tests/test_math.cpp)
    target_include_directories(fsmlib_test_math PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(fsmlib_test_math fsmlib)
    add_test(fsmlib_test_math_run fsmlib_test_math)

    # Add the test.
    add_executable(fsmlib_test_linalg ${PROJECT_SOURCE_DIR}/tests/test_linalg.cpp)
    target_include_directories(fsmlib_test_linalg PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(fsmlib_test_linalg fsmlib)
    add_test(fsmlib_test_linalg_run fsmlib_test_linalg)

    # Add the test.
    add_executable(fsmlib_test_control ${PROJECT_SOURCE_DIR}/tests/test_control.cpp)
    target_include_directories(fsmlib_test_control PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(fsmlib_test_control fsmlib)
    add_test(fsmlib_test_control_run fsmlib_test_control)

endif()

# -----------------------------------------------------------------------------
# DOCUMENTATION
# -----------------------------------------------------------------------------

if(DOXYGEN_FOUND)

    # Record the options that describe how to populate the specified content.
    FetchContent_Declare(
        doxygenawesome
        GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css
        GIT_TAG main
    )
    # Retrieve the properties related to the content.
    FetchContent_GetProperties(doxygenawesome)
    # If not populated, make the content available.
    if(NOT doxygenawesome_POPULATED)
        message(STATUS "Retrieving `doxygen-awesome-css`...")
        # Ensures the named dependencies have been populated.
        FetchContent_MakeAvailable(doxygenawesome)
        # Hide fetchcontent variables, otherwise with ccmake it's a mess.
        mark_as_advanced(FORCE
            FETCHCONTENT_UPDATES_DISCONNECTED_DOXYGENAWESOME
            FETCHCONTENT_SOURCE_DIR_DOXYGENAWESOME
        )
    endif()

    # = CUSTOMIZATION =========================================================
    set(DOXYGEN_WARN_FORMAT "$file:$line: $text")
    set(DOXYGEN_PROJECT_NAME fsmlib)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${PROJECT_SOURCE_DIR}/README.md)
    set(DOXYGEN_SHOW_INCLUDE_FILES NO)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_HTML_HEADER ${doxygenawesome_SOURCE_DIR}/doxygen-custom/header.html)
    set(DOXYGEN_HTML_EXTRA_STYLESHEET ${doxygenawesome_SOURCE_DIR}/doxygen-awesome.css)
    set(DOXYGEN_HTML_EXTRA_FILES
        ${doxygenawesome_SOURCE_DIR}/doxygen-awesome-fragment-copy-button.js
        ${doxygenawesome_SOURCE_DIR}/doxygen-awesome-paragraph-link.js
        ${doxygenawesome_SOURCE_DIR}/doxygen-awesome-darkmode-toggle.js
    )
    doxygen_add_docs(
        fsmlib_documentation
        ${PROJECT_SOURCE_DIR}/README.md
        ${PROJECT_SOURCE_DIR}/LICENSE.md
        ${PROJECT_SOURCE_DIR}/include/fsmlib/control.hpp
        ${PROJECT_SOURCE_DIR}/include/fsmlib/feq.hpp
        ${PROJECT_SOURCE_DIR}/include/fsmlib/fsmlib.hpp
        ${PROJECT_SOURCE_DIR}/include/fsmlib/io.hpp
        ${PROJECT_SOURCE_DIR}/include/fsmlib/linalg.hpp
        ${PROJECT_SOURCE_DIR}/include/fsmlib/math.hpp
        ${PROJECT_SOURCE_DIR}/include/fsmlib/traits.hpp
        ${PROJECT_SOURCE_DIR}/include/fsmlib/view.hpp
    )
endif()
