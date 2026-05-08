ARC_TOOLCHAIN ?= gnu
ifeq ($(ARC_TOOLCHAIN), gnu)
  CC  := arc-elf32-gcc
  CXX := arc-elf32-g++
  AR  := arc-elf32-ar
  LD  := arc-elf32-ld
endif

TFLM_DIR  := ../himax_tflm
SDK_DIR   := $(TFLM_DIR)/himax_we1_sdk

CXXFLAGS := \
  -fno-rtti -fpermissive -fno-exceptions \
  -fno-threadsafe-statics -fno-unwind-tables \
  -ffunction-sections -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  -O3 -Wno-unused-parameter \
  -DREDUCE_CODESIZE -mxy \
  -include $(SDK_DIR)/core_config.h \
  -mcpu=em4_fpus -mlittle-endian \
  -mcode-density -mdiv-rem -mswap -mnorm \
  -mmpy-option=6 -mbarrel-shifter -mfpu=fpus_all \
  -I. \
  -I$(TFLM_DIR) \
  -I$(SDK_DIR) \
  -I$(TFLM_DIR)/third_party/gemmlowp \
  -I$(TFLM_DIR)/third_party/flatbuffers/include \
  -I$(TFLM_DIR)/third_party/ruy \
  -I$(TFLM_DIR)/third_party/arc_mli_package/include \
  -I$(TFLM_DIR)/third_party/arc_mli_package/include/api \
  -DSCRATCH_MEM_Z_SIZE=0x10000 \
  -I$(TFLM_DIR)/third_party/kissfft

LDFLAGS := \
  -Wl,-marcv2elfx \
  -Wl,--strip-debug \
  -Wl,--gc-sections \
  -Wl,-Map=gesture.map \
  -Wl,--start-group \
    $(SDK_DIR)/libcpuarc.a \
    $(SDK_DIR)/libbss.a \
    $(SDK_DIR)/libboard_socket.a \
    $(SDK_DIR)/libboard_open_socket.a \
    $(SDK_DIR)/liblibcommon.a \
    $(SDK_DIR)/liblibsensordp.a \
    $(SDK_DIR)/liblibclib.a \
    $(SDK_DIR)/liblibtflm.a \
  -Wl,--end-group \
  -L$(TFLM_DIR)/third_party/arc_mli_package/bin/himax_arcem9d_r16/release \
  -Wl,-lmli \
  -L$(TFLM_DIR)/third_party/mw_gnu_dependencies/gnu_depend_lib \
  -Wl,-lmwdepend

# Only include ops used by our model - much smaller than AllOpsResolver
TFLM_OBJS := \
  $(TFLM_DIR)/tensorflow/lite/micro/himax_we1_evb/debug_log.o \
  $(TFLM_DIR)/tensorflow/lite/micro/flatbuffer_utils.o \
  $(TFLM_DIR)/tensorflow/lite/micro/memory_helpers.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_allocator.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_error_reporter.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_graph.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_interpreter.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_profiler.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_resource_variable.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_string.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_time.o \
  $(TFLM_DIR)/tensorflow/lite/micro/micro_utils.o \
  $(TFLM_DIR)/tensorflow/lite/micro/mock_micro_graph.o \
  $(TFLM_DIR)/tensorflow/lite/micro/recording_micro_allocator.o \
  $(TFLM_DIR)/tensorflow/lite/micro/recording_simple_memory_allocator.o \
  $(TFLM_DIR)/tensorflow/lite/micro/simple_memory_allocator.o \
  $(TFLM_DIR)/tensorflow/lite/micro/system_setup.o \
  $(TFLM_DIR)/tensorflow/lite/micro/memory_planner/greedy_memory_planner.o \
  $(TFLM_DIR)/tensorflow/lite/micro/memory_planner/linear_memory_planner.o \
  $(TFLM_DIR)/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.o \
  $(TFLM_DIR)/tensorflow/lite/c/common.o \
  $(TFLM_DIR)/tensorflow/lite/core/api/error_reporter.o \
  $(TFLM_DIR)/tensorflow/lite/core/api/flatbuffer_conversions.o \
  $(TFLM_DIR)/tensorflow/lite/core/api/op_resolver.o \
  $(TFLM_DIR)/tensorflow/lite/core/api/tensor_utils.o \
  $(TFLM_DIR)/tensorflow/lite/kernels/internal/quantization_util.o \
  $(TFLM_DIR)/tensorflow/lite/kernels/kernel_util.o \
  $(TFLM_DIR)/tensorflow/lite/schema/schema_utils.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/conv_common.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/depthwise_conv_common.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/fully_connected_common.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/softmax_common.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/softmax.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/reshape.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/reduce.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/add_common.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/quantize.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/quantize_common.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/dequantize.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/dequantize_common.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/pooling_common.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/conv.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/depthwise_conv.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/fully_connected.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/pooling.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/mli_interface.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/mli_slicers.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/arc_mli/add.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/kernel_util.o \
  $(TFLM_DIR)/tensorflow/lite/micro/kernels/logistic_common.o

SRCS := main_inference.cc capture.cc
OBJS := $(SRCS:.cc=.o)

all: gesture.elf

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

gesture.elf: $(OBJS) $(TFLM_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(TFLM_OBJS) $(LDFLAGS)

flash: gesture.elf
	cp gesture.elf $(TFLM_DIR)/image_gen_linux/
	cd $(TFLM_DIR)/image_gen_linux && PATH=$$PWD:$$PATH && \
	./image_gen_gnu -e gesture.elf -s 1024 -o gesture.img && \
	cp gesture*.img ../..

clean:
	rm -f $(OBJS) gesture.elf gesture.map ../../gesture*.img
