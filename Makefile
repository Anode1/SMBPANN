#
# Copyright (C) 2001 Vasili Gavrilov
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or any later version.
#
# SMBPANN in C99. Layout and build follow the AIS project: no framework, a plain
# Makefile and cc, sources auto-globbed so dropping in a new *.c just compiles.
#
# Honors the standard variables so packagers and other compilers just work:
#   CC CFLAGS CPPFLAGS LDFLAGS LDLIBS  -- project flags are APPENDED, not
#   overriding yours. Profiles: make | make release | make debug | make pedantic
#

SHELL = /bin/sh

BIN     = smbpann
EVOLVE  = evolve
GENTASK = gentask
TESTBIN = smbpann_ut

# -- toolchain (overridable on the command line or by the packager) --------
CC       ?= cc
CFLAGS   ?= -O2
CPPFLAGS ?=
LDFLAGS  ?=
LDLIBS   ?=

# -- project-required flags (always applied, before yours) -----------------
SMB_STD    = -std=c99
SMB_WARN   = -W -Wall
SMB_CFLAGS = $(SMB_STD) $(SMB_WARN)
SMB_MATH   = -lm            # act.c uses exp()

# Version: single source = the git tag, like AIS. `git describe` gives the exact
# tag on a release build and tag+commits+sha between tags; a bare copy with no
# .git falls back to 0.0.0-dev. Stamped into the binary via -D (main.c reads it).
SMB_VERSION := $(shell git describe --tags --always --dirty 2>/dev/null | sed 's/^v//')
ifeq ($(strip $(SMB_VERSION)),)
SMB_VERSION := 0.0.0-dev
endif
SMB_VERDEF = -DSMB_VERSION='"$(SMB_VERSION)"'

# Sources = every *.c at the top level. Three programs share the engine objects:
# smbpann (worker, main.o), evolve (search, evolve.o), gentask (task generator,
# gentask.o). Each links every object EXCEPT the other programs' mains. tests.o
# is a harmless empty unit in the non-test build.
SOURCES.c := $(wildcard *.c)
OBJS       = $(SOURCES.c:.c=.o)
SMBPANN_OBJS = $(filter-out evolve.o gentask.o, $(OBJS))
EVOLVE_OBJS  = $(filter-out main.o gentask.o,   $(OBJS))
GENTASK_OBJS = $(filter-out main.o evolve.o,    $(OBJS))

# -- profiles tweak only the standard knobs --------------------------------
debug    : CFLAGS  = -g -O0
debug    : SMB_WARN += -Wundef
pedantic : SMB_STD  = -std=c99 -pedantic
pedantic : SMB_WARN += -Wundef -Wstrict-prototypes -Wmissing-prototypes -Wmissing-declarations
release  : CFLAGS   = -O2

.SUFFIXES:
.SUFFIXES: .d .o .h .c
%.o: %.c
	$(CC) $(SMB_CFLAGS) $(SMB_VERDEF) $(CPPFLAGS) $(CFLAGS) -MMD -c $< -o $@

.PHONY: all release debug pedantic clean ut ut-asan ut-ubsan check bbtest modnas modevo conv2d nasxover nb101_extract nb101 conv_emerge emerge_tie

all release debug pedantic: $(BIN) $(EVOLVE) $(GENTASK)

# bbtest / modnas: standalone validation probes (validation/*.c), separate from
# the engine (own PRNG, own main). bbtest = building-block crossover on trap
# functions; modnas = a parallel-branch net (gradient-checked). See the paper.
bbtest:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o bbtest validation/bbtest.c
modnas:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o modnas validation/modular_nas.c $(SMB_MATH)
nasxover:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o nasxover validation/nasxover.c $(SMB_MATH)
nb101_extract:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o nb101_extract validation/nb101_extract.c
emerge_tie:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_tie validation/emerge_tie.c $(SMB_MATH)
emerge_local:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_local validation/emerge_local.c $(SMB_MATH)
emerge_compose:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_compose validation/emerge_compose.c $(SMB_MATH)
emerge_arch:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_arch validation/emerge_arch.c $(SMB_MATH)
emerge_twoop:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_twoop validation/emerge_twoop.c $(SMB_MATH)
emerge_staged:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_staged validation/emerge_staged.c $(SMB_MATH)
conv_emerge:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o conv_emerge validation/conv_emerge.c $(SMB_MATH)
nb101:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o nb101 validation/nb101.c $(SMB_MATH)
conv2d:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o conv2d validation/conv2d.c $(SMB_MATH)
modevo:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o modevo validation/modevo.c $(SMB_MATH)

$(BIN): $(SMBPANN_OBJS)
	$(CC) $(SMB_CFLAGS) $(CFLAGS) $(LDFLAGS) -o $(BIN) $(SMBPANN_OBJS) $(LDLIBS) $(SMB_MATH)

$(EVOLVE): $(EVOLVE_OBJS)
	$(CC) $(SMB_CFLAGS) $(CFLAGS) $(LDFLAGS) -o $(EVOLVE) $(EVOLVE_OBJS) $(LDLIBS) $(SMB_MATH)

$(GENTASK): $(GENTASK_OBJS)
	$(CC) $(SMB_CFLAGS) $(CFLAGS) $(LDFLAGS) -o $(GENTASK) $(GENTASK_OBJS) $(LDLIBS) $(SMB_MATH)

# Test binary: all sources with -DUNIT_TEST (which compiles out main.c's main()
# so tests.c provides the entry point). .PHONY ut/check always run.
$(TESTBIN): $(SOURCES.c)
	$(CC) $(SMB_CFLAGS) -g -DUNIT_TEST $(CPPFLAGS) $(SOURCES.c) -o $(TESTBIN) $(LDLIBS) $(SMB_MATH)

ut check: $(TESTBIN)
	./$(TESTBIN)

# ut-asan / ut-ubsan: the same tests under AddressSanitizer / UBSan, so memory
# errors and undefined behaviour abort with a file:line report instead of
# passing silently under -O2. Kept out of the default build (2-3x slower).
ut-asan:
	$(CC) $(SMB_CFLAGS) -g -DUNIT_TEST -fsanitize=address -fno-omit-frame-pointer \
	  $(CPPFLAGS) $(SOURCES.c) -o $(TESTBIN)_asan $(LDLIBS) $(SMB_MATH)
	./$(TESTBIN)_asan
ut-ubsan:
	$(CC) $(SMB_CFLAGS) -g -DUNIT_TEST -fsanitize=undefined -fno-sanitize-recover=undefined \
	  $(CPPFLAGS) $(SOURCES.c) -o $(TESTBIN)_ubsan $(LDLIBS) $(SMB_MATH)
	./$(TESTBIN)_ubsan

clean:
	-rm -f $(BIN) $(EVOLVE) $(GENTASK) $(TESTBIN) $(TESTBIN)_asan $(TESTBIN)_ubsan bbtest modnas modevo conv2d nasxover nb101_extract nb101 conv_emerge emerge_tie $(OBJS) $(OBJS:.o=.d)

-include $(OBJS:.o=.d)
emerge_2d:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_2d validation/emerge_2d.c $(SMB_MATH)
emerge_2d_orient:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_2d_orient validation/emerge_2d_orient.c $(SMB_MATH)
emerge_2d_chan:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_2d_chan validation/emerge_2d_chan.c $(SMB_MATH)
emerge_2d_grow:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_2d_grow validation/emerge_2d_grow.c $(SMB_MATH)
emerge_2d_deep:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_2d_deep validation/emerge_2d_deep.c $(SMB_MATH)
emerge_2d_deep2:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_2d_deep2 validation/emerge_2d_deep2.c $(SMB_MATH)
emerge_translate:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_translate validation/emerge_translate.c $(SMB_MATH)
emerge_2d_compete:
	$(CC) $(SMB_CFLAGS) $(CFLAGS) -o emerge_2d_compete validation/emerge_2d_compete.c $(SMB_MATH)
