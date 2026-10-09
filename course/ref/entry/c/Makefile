# c/Makefile: builds libtinyllm (DESIGN 2.15). Your file; this is the reference
# for course CI. `make -C c` writes build/libtinyllm.a (linked into the engine
# by tl-sys) and build/libtinyllm.{dylib,so} (loaded by Python through ctypes).
# SANITIZE=1 adds ASan and UBSan.

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -fPIC
CPPFLAGS += -I../contracts/c/include
ifeq ($(SANITIZE),1)
CFLAGS  += -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS += -fsanitize=address,undefined
endif

SRCS := $(sort $(wildcard src/*/*.c))
OBJS := $(patsubst src/%.c,build/obj/%.o,$(SRCS))
SHARED_EXT := $(if $(filter Darwin,$(shell uname -s)),dylib,so)

.PHONY: all clean
all: build/libtinyllm.a build/libtinyllm.$(SHARED_EXT)

build/obj/%.o: src/%.c $(wildcard ../contracts/c/include/*.h ../contracts/c/include/tinyllm/*.h)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/libtinyllm.a: $(OBJS)
	$(AR) rcs $@ $^

build/libtinyllm.$(SHARED_EXT): $(OBJS)
	$(CC) -shared $(LDFLAGS) $^ -o $@ -lm

clean:
	rm -rf build
