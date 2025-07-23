cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

find_package(TBLIS REQUIRED)


include(FetchContent)
FetchContent_Declare(
  tblis
  GIT_REPOSITORY https://github.com/MatthewsResearchGroup/tblis.git
  GIT_TAG 88b5cc5e610bf608a274a34053883af66da43930
  FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(tblis)


nanobind_add_module(pytblis pytblis.cxx)
set_target_properties(pytblis PROPERTIES CXX_STANDARD 20)

target_link_libraries(pytblis PRIVATE TBLIS::tblis)

install(TARGETS pytblis LIBRARY DESTINATION .)
