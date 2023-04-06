# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_TIM_VX_CODEGEN)
    if(NOT DEFINED TVM_LLVM_VERSION)
        message(FATAL_ERROR "Support for offloading to TIM-VX requires LLVM support.")
    endif()

    set(TIM_VX_PATH ${PROJECT_SOURCE_DIR}/3rdparty/tim-vx)
    # Detect custom TIM-VX path.
    if (NOT USE_TIM_VX_CODEGEN STREQUAL "ON")
        set(TIM_VX_PATH ${USE_TIM_VX_CODEGEN})
    endif()

    set(TIM_VX_INCLUDE_DIRS ${TIM_VX_PATH}/include)
    include_directories(SYSTEM ${TIM_VX_INCLUDE_DIRS})

    find_library(TIM_VX_LIBS
        NAMES tim-vx
        HINTS "${TIM_VX_PATH}/lib" "${TIM_VX_PATH}/build"
        REQUIRED
    )

    list(APPEND TVM_LINKER_LIBS ${TIM_VX_LIBS})
    message(STATUS "Build with TIM-VX codegen: ${TIM_VX_LIBS}")

    tvm_file_glob(GLOB TIM_VX_COMPILER_SRCS "src/relay/backend/contrib/tim_vx/*.cc")
    list(APPEND COMPILER_SRCS ${TIM_VX_COMPILER_SRCS})
endif()

if(USE_TIM_VX_RUNTIME)
    set(TIM_VX_PATH ${PROJECT_SOURCE_DIR}/3rdparty/tim-vx)
    # Detect custom TIM-VX path.
    if (NOT USE_TIM_VX_RUNTIME STREQUAL "ON")
        set(TIM_VX_PATH ${USE_TIM_VX_RUNTIME})
    endif()

    set(TIM_VX_INCLUDE_DIRS ${TIM_VX_PATH}/include)
    include_directories(SYSTEM ${TIM_VX_INCLUDE_DIRS})

    find_library(TIM_VX_LIBS
        NAMES tim-vx
        HINTS "${TIM_VX_PATH}/lib" "${TIM_VX_PATH}/build"
        REQUIRED
    )

    list(APPEND TVM_RUNTIME_LINKER_LIBS ${TIM_VX_LIBS})
    message(STATUS "Build with TIM-VX runtime: ${TIM_VX_LIBS}")

    tvm_file_glob(GLOB TIM_VX_RUNTIME_SRCS "src/runtime/contrib/tim_vx/tim_vx_*.cc")
    list(APPEND RUNTIME_SRCS ${TIM_VX_RUNTIME_SRCS})
    list(APPEND RUNTIME_SRCS "src/runtime/contrib/tim_vx/nbg_parser.cc")

    # Set TIM-VX runtime enabled flag.
    add_definitions(-DTVM_GRAPH_EXECUTOR_TIM_VX)
elseif(USE_OPENVX_RUNTIME)
    set(OVX_PATH ${PROJECT_SOURCE_DIR}/3rdparty/OVX)
    # Detect custom OpenVX runtime path.
    if (NOT USE_OPENVX_RUNTIME STREQUAL "ON")
        set(OVX_PATH ${USE_OPENVX_RUNTIME})
    endif()

    set(OVX_INCLUDE_DIRS ${OVX_PATH}/include)
    include_directories(${OVX_INCLUDE_DIRS})

    find_library(OVX_LIBS
        NAMES OpenVX
        HINTS "${OVX_PATH}/lib" "${OVX_PATH}/drivers"
        REQUIRED
    )

    list(APPEND TVM_RUNTIME_LINKER_LIBS ${OVX_LIBS})
    message(STATUS "Build with OpenVX runtime: ${OVX_LIBS}")

    tvm_file_glob(GLOB OVX_RUNTIME_SRCS "src/runtime/contrib/tim_vx/ovx_*.cc")
    list(APPEND RUNTIME_SRCS ${OVX_RUNTIME_SRCS})
    list(APPEND RUNTIME_SRCS "src/runtime/contrib/tim_vx/nbg_parser.cc")

    # Set OpenVX runtime enabled flag.
    add_definitions(-DTVM_GRAPH_EXECUTOR_OPENVX)
elseif(USE_VIPLITE_RUNTIME)
    set(VIPLITE_PATH ${PROJECT_SOURCE_DIR}/3rdparty/viplite)
    # Detect custom VIPLite driver path.
    if (NOT USE_VIPLITE_RUNTIME STREQUAL "ON")
        set(VIPLITE_PATH ${USE_VIPLITE_RUNTIME})
    endif()

    set(VIPLITE_INCLUDE_DIRS ${VIPLITE_PATH}/include)
    include_directories(${VIPLITE_INCLUDE_DIRS})

    find_library(VIPLITE_LIBS
        NAMES VIPlite VIPuser
        HINTS "${VIPLITE_PATH}/lib" "${VIPLITE_PATH}/drivers"
        REQUIRED
    )

    list(APPEND TVM_RUNTIME_LINKER_LIBS ${VIPLITE_LIBS})
    message(STATUS "Build with VIPLite runtime: ${VIPLITE_LIBS}")

    tvm_file_glob(GLOB VIPLITE_RUNTIME_SRCS "src/runtime/contrib/tim_vx/viplite_*.cc")
    list(APPEND RUNTIME_SRCS ${VIPLITE_RUNTIME_SRCS})
endif()