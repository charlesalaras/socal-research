ROCM_PATH ?= /opt/rocm
ROCBLAS_PATH ?= $(ROCM_PATH)/rocblas
DAGEE_PATH ?= $(CURDIR)/../DAGEE
ATMI_PATH ?= $(ROCM_PATH)/atmi
RSMI_PATH ?= /opt/rocm/rocm_smi

DEBUG ?= 0
TIME  ?= 1
POWER ?= 1

CXX = $(ROCM_PATH)/bin/hipcc
INC = -Iinclude -I$(ROCM_PATH)/include -I$(ROCBLAS_PATH)/include
DAGEE_INC = -I$(DAGEE_PATH)/DAGEE-lib/include -I$(DAGEE_PATH)/cppUtils/include
ATMI_INC = -I$(ATMI_PATH)/include
RSMI_INC = -I${RSMI_PATH}/include


CXXFLAGS =  $(INC)
LDFLAGS = -L$(ROCBLAS_PATH)/lib/ -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib

ATMI_LDFLAGS = -L$(ATMI_PATH)/lib -latmi_runtime -Wl,-rpath=$(ATMI_PATH)/lib
RSMI_LDFLAGS = -L$(RSMI_PATH)/lib -lrocm_smi64 -Wl,-rpath=$(RSMI_PATH)/lib

ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG
endif

ifeq ($(TIME), 1)
	CXXFLAGS += -DTIME
endif

ifeq ($(POWER), 1)
	CXXFLAGS += -DPOWER
endif

all: saxpy saxpy_dagee

saxpy:
	$(CXX) $(CXXFLAGS) saxpy.hip.cpp $(LDFLAGS) -o saxpy

saxpy_dagee:
	$(CXX) $(CXXFLAGS) saxpy_dagee.hip.cpp $(LDFLAGS) -o saxpy_dagee

clean:
	rm saxpy saxpy_dagee
