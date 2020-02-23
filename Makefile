PROJ=Convolution

CC=gcc

CFLAGS=-std=c99 -Wall -DUNIX

# Check for 32-bit vs 64-bit
PROC_TYPE = $(strip $(shell uname -m | grep 64))
 
# Check for Mac OS
OS = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN = $(strip $(findstring DARWIN, $(OS)))

# MacOS System
ifneq ($(DARWIN),)
	CFLAGS += -DMAC
	LIBS=-framework OpenCL

	ifeq ($(PROC_TYPE),)
		CFLAGS+=-arch i386
	else
		CFLAGS+=-arch x86_64
	endif
else
	# Linux OS
	LIBS=-lOpenCL
	ifeq ($(PROC_TYPE),)
		CFLAGS+=-m32
	else
		CFLAGS+=-m64
	endif
endif

# Check for Linux-AMD
ifdef AMDAPPSDKROOT
   	INC_DIRS=. $(AMDAPPSDKROOT)/include
	ifeq ($(PROC_TYPE),)
		LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86
	else
		LIB_DIRS=$(AMDAPPSDKROOT)/lib/x86_64
	endif
else
	# Check for Linux-Nvidia
	ifdef NVSDKCOMPUTE_ROOT
		INC_DIRS=. $(NVSDKCOMPUTE_ROOT)/OpenCL/common/inc
	endif
endif 

ifeq ($(ver), debug)
	CFLAGS += -g -DDEBUG
endif

$(PROJ): $(PROJ).c
	$(CC) $(CFLAGS) -o bin/$@ $^ $(INC_DIRS:%=-I%) $(LIB_DIRS:%=-L%) $(LIBS)
	cp Convolution.cl bin/

.PHONY: clean

clean:
	rm bin/$(PROJ)
