#
# SMBPANN (Ada) -- the active implementation. GNAT + gnatmake, no gprbuild
# needed, mirroring the plain-Makefile spirit of AIS's C build.
#
#   make        build every program (all warnings + contracts enabled)
#   make run    build and run the XOR demo
#   make test   build and run the unit tests (arena, ...)
#   make clean  remove build artifacts
#
# -gnatwa   : every warning (a warning is a defect, as in AIS)
# -gnata    : enable Pre/Post/Assert contracts as runtime checks
# -gnat2012 : the language level used across the engine
#

GNATMAKE ?= gnatmake
ADAFLAGS  = -gnatwa -gnata -gnat2012

PROGRAMS = xor_demo arena_test

.PHONY: all run test clean $(PROGRAMS)

all: $(PROGRAMS)

$(PROGRAMS):
	$(GNATMAKE) $(ADAFLAGS) $@.adb

run: xor_demo
	./xor_demo

test: arena_test
	./arena_test

clean:
	-rm -f *.o *.ali $(PROGRAMS)
