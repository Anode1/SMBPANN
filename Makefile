#
# SMBPANN (Ada) -- the active implementation. GNAT + gnatmake, no gprbuild
# needed, mirroring the plain-Makefile spirit of AIS's C build.
#
#   make        build the demo (all warnings + contracts enabled)
#   make run    build and run the XOR demo
#   make clean  remove build artifacts
#
# -gnatwa : every warning (a warning is a defect, as in AIS)
# -gnata  : enable Pre/Post/Assert contracts as runtime checks
# -gnat2012 : the language level used across the engine
#

GNATMAKE ?= gnatmake
ADAFLAGS  = -gnatwa -gnata -gnat2012

MAIN = xor_demo

.PHONY: all run clean

all: $(MAIN)

$(MAIN): $(wildcard *.ad[sb])
	$(GNATMAKE) $(ADAFLAGS) $(MAIN).adb

run: all
	./$(MAIN)

clean:
	-rm -f *.o *.ali $(MAIN)
