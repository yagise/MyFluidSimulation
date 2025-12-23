
cmake_minimum_required(VERSION 3.20)

project(FluidSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

option(FLUIDSIM_BUILD_GUI "Build OpenGL GUI viewer (fluidsim_compare)" ON)

# GLM は voxelize 等でも使うため、GUI の有無に関わらず必須
find_package(glm CONFIG REQUIRED)

# GUI をビルドする場合のみ OpenGL 関連を要求する
if(FLUIDSIM_BUILD_GUI)
  find_package(glfw3 REQUIRED)
  find_package(glad CONFIG REQUIRED)
endif()
find_package(CUDAToolkit REQUIRED)

if(FLUIDSIM_BUILD_GUI)
  add_compile_definitions(GLFW_INCLUDE_NONE)
endif()

add_library(lbm3d_obj OBJECT
  src/lbm/lbm3d_legacy.cu
  src/lbm/lbm3d_legacy.hpp
  src/lbm/lbm_reinit.cu
)
set_target_properties(lbm3d_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(lbm3d_obj PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_library(lbm3d_home_obj OBJECT
  src/lbm/lbm3d_home.cu
  src/lbm/lbm3d_home.hpp
)
set_target_properties(lbm3d_home_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(lbm3d_home_obj PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_library(lbm3d_hybrid_obj OBJECT
  src/lbm/lbm3d_hybrid.cu
  src/lbm/lbm3d_hybrid.hpp
)
set_target_properties(lbm3d_hybrid_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(lbm3d_hybrid_obj PUBLIC ${CMAKE_SOURCE_DIR}/src)


add_library(moment_vtk_writer STATIC
  src/moment_vtk_writer.cu
)
target_include_directories(moment_vtk_writer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(moment_vtk_writer PUBLIC USE_CUDA)
target_link_libraries(moment_vtk_writer PUBLIC CUDA::cudart)

if(FLUIDSIM_BUILD_GUI)
  add_library(vis3d STATIC
    src/render/Vis3D.cpp
    src/render/Vis3D.hpp
    src/render/MeshRenderer.cpp
    src/render/MeshRenderer.hpp
  )
  target_link_libraries(vis3d PUBLIC glad::glad glm::glm)
  target_include_directories(vis3d PUBLIC ${CMAKE_SOURCE_DIR}/src)

  add_library(core STATIC
    src/render/window.cpp
    src/render/window.hpp
  )
  target_link_libraries(core PUBLIC glfw)
  target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src)

  add_executable(fluidsim_compare
    src/main_compare.cpp
    src/cuda_dummy.cu
    src/geom/stl_loader.cpp
    src/geom/voxelize.cpp
    src/fields/field_ops.cu
    src/fields/init_profiles.cu
  )
  set_target_properties(fluidsim_compare PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
  )
  target_sources(fluidsim_compare PRIVATE
    $<TARGET_OBJECTS:lbm3d_obj>
    $<TARGET_OBJECTS:lbm3d_home_obj>
    $<TARGET_OBJECTS:lbm3d_hybrid_obj>
  )
  target_link_libraries(fluidsim_compare PRIVATE
    core vis3d glfw glad::glad glm::glm CUDA::cudart
    moment_vtk_writer
  )
endif()

# -----------------------------------------------------------------------------
# 論文用: 手法ごとの headless 実行バイナリ
#   - fluidsim_legacy : 従来法(BGK)
#   - fluidsim_home   : HOME 法
#   - fluidsim_hybrid : 従来法 + HOME(B0)
#
# 目的:
#   - 1つの巨大な main に全部が混ざる状態を避け、論文に載せやすい形にする。
#   - GUI(OpenGL) 依存を外して、計算・出力に集中する。
# -----------------------------------------------------------------------------

add_executable(fluidsim_legacy
  src/apps/main_legacy.cpp
  src/cuda_dummy.cu
  src/geom/stl_loader.cpp
  src/geom/voxelize.cpp
  src/fields/field_ops.cu
  src/fields/init_profiles.cu
)
set_target_properties(fluidsim_legacy PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared
)
target_sources(fluidsim_legacy PRIVATE
  $<TARGET_OBJECTS:lbm3d_obj>
)
target_link_libraries(fluidsim_legacy PRIVATE CUDA::cudart moment_vtk_writer glm::glm)
target_include_directories(fluidsim_legacy PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(fluidsim_home
  src/apps/main_home.cpp
  src/cuda_dummy.cu
  src/geom/stl_loader.cpp
  src/geom/voxelize.cpp
  src/fields/field_ops.cu
  src/fields/init_profiles.cu
)
set_target_properties(fluidsim_home PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared
)
target_sources(fluidsim_home PRIVATE
  $<TARGET_OBJECTS:lbm3d_home_obj>
  $<TARGET_OBJECTS:lbm3d_obj>   # reinit_equilibrium_from_macro を提供
)
target_link_libraries(fluidsim_home PRIVATE CUDA::cudart moment_vtk_writer glm::glm)
target_include_directories(fluidsim_home PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(fluidsim_hybrid
  src/apps/main_hybrid.cpp
  src/cuda_dummy.cu
  src/geom/stl_loader.cpp
  src/geom/voxelize.cpp
  src/fields/field_ops.cu
  src/fields/init_profiles.cu
)
set_target_properties(fluidsim_hybrid PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared
)
target_sources(fluidsim_hybrid PRIVATE
  $<TARGET_OBJECTS:lbm3d_hybrid_obj>
  $<TARGET_OBJECTS:lbm3d_obj>   # reinit_equilibrium_from_macro を提供
)
target_link_libraries(fluidsim_hybrid PRIVATE CUDA::cudart moment_vtk_writer glm::glm)
target_include_directories(fluidsim_hybrid PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(fluidsim_bench
  src/bench/main_bench_experiments.cpp
  src/bench/bench_reduce.cu
  src/cuda_dummy.cu
)
set_target_properties(fluidsim_bench PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared
)
target_sources(fluidsim_bench PRIVATE
  $<TARGET_OBJECTS:lbm3d_obj>
  $<TARGET_OBJECTS:lbm3d_home_obj>
  $<TARGET_OBJECTS:lbm3d_hybrid_obj>
)
target_link_libraries(fluidsim_bench PRIVATE CUDA::cudart moment_vtk_writer)
target_include_directories(fluidsim_bench PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(fluidsim_tgv
  src/bench/main_taylor_green.cpp
  src/cuda_dummy.cu
  src/fields/field_ops.cu
)
set_target_properties(fluidsim_tgv PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_RUNTIME_LIBRARY Shared
)
target_sources(fluidsim_tgv PRIVATE
  $<TARGET_OBJECTS:lbm3d_obj>
  $<TARGET_OBJECTS:lbm3d_home_obj>
  $<TARGET_OBJECTS:lbm3d_hybrid_obj>
)
target_link_libraries(fluidsim_tgv PRIVATE CUDA::cudart moment_vtk_writer)
target_include_directories(fluidsim_tgv PRIVATE ${CMAKE_SOURCE_DIR}/src)
