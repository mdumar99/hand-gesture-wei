ARC_TOOLCHAIN ?= gnu
ifeq ($(ARC_TOOLCHAIN), gnu)
  CC  := arc-elf32-gcc
  CXX := arc-elf32-g++
  AR  := arc-elf32-ar
  LD  := arc-elf32-ld
endif

TFLM_DIR  := ../himax_tflm
SDK_DIR   := $(TFLM_DIR)/himax_we1_sdk
GEN_TOOL  := $(TFLM_DIR)/image_gen_linux/image_gen_gnu

CXXFLAGS := \
  -fno-rtti -fpermissive -fno-exceptions \
  -fno-threadsafe-statics -fno-unwind-tables \
  -ffunction-sections -fdata-sections \
  -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY \
  -DTF_LITE_DISABLE_X86_NEON \
  -O3 -Wall -Wextra \
  -DREDUCE_CODESIZE -mxy \
  -include $(SDK_DIR)/core_config.h \
  -mcpu=em4_fpus -mlittle-endian \
  -mcode-density -mdiv-rem -mswap -mnorm \
  -mmpy-option=6 -mbarrel-shifter -mfpu=fpus_all \
  -I$(TFLM_DIR) \
  -I$(SDK_DIR)

LDFLAGS := \
  -Wl,-marcv2elfx \
  -Wl,--strip-debug \
  -Wl,--gc-sections \
  -Wl,-Map=capture.map \
  -L$(SDK_DIR) \
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
  -L../himax_tflm/third_party/mw_gnu_dependencies/gnu_depend_lib \
  -Wl,-lmwdepend

SRCS := main.cc capture.cc
OBJS := $(SRCS:.cc=.o)

all: capture.elf

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

capture.elf: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

flash: capture.elf
	cp capture.elf $(TFLM_DIR)/image_gen_linux/
	cd $(TFLM_DIR)/image_gen_linux && PATH=$$PWD:$$PATH && \
	./image_gen_gnu -e capture.elf -s 1024 -o capture.img && \
	cp capture*.img ../..

clean:
	rm -f $(OBJS) capture.elf capture.map ../../capture*.img
