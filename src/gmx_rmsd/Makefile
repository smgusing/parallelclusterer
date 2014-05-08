# Generated automatically from Makefile.in by configure.
#
# This is a Gromacs 4.5 template makefile for your own utility programs.
#
# Copy this file to whatever directory you are using for your own
# software and add more targets like the template one below.
#
# If you are using gmake it is relatively straightforward to add
# an include based on environment variables (like previous Gromacs versions)
# to select compiler flags and stuff automatically, but below it is static:
#

# Variables set by the configuration script:
LIBS         = -lmd -lgmx -lfftw3f   -lnsl -lm -lgomp

LDFLAGS      = -L/home/alex/gromacs/gmx455/lib   
CFLAGS	     = -O3 -fopenmp -fomit-frame-pointer -finline-functions -Wall -Wno-unused -msse2 -funroll-all-loops -std=gnu99 -fexcess-precision=fast -pthread -I./include  -I/home/alex/gromacs/gmx455/include/gromacs

#LDFLAGS      = -L/storage/gurpreet/install/gmx455/lib -L/storage/gurpreet/install/openmpi-1.4.1/lib  -L/storage/gurpreet/install/fftw-3.2.1/lib  
#CFLAGS	     = -O3 -fopenmp  -fomit-frame-pointer -finline-functions -Wall -Wno-unused -msse2 -funroll-all-loops -std=gnu99 -pthread -I./include  -I/storage/gurpreet/install/gmx455/include/gromacs


CC           = cc 
LD           = $(CC)
SRC	     = $(wildcard *.c)
OBJS	     = $(patsubst %.c,%.o,$(SRC))
# The real make targets - note that most make programs support
# the shortcut $^ instead of listing all object files a second
# time, but we cannot count on it...



libcmetric.so: $(OBJS)
		$(LD) $(LDFLAGS) -shared -Wl,-soname,libcmetric.so.1 -o $@ $^ $(LIBS) -lc


%.o: %.c
	    $(LD) -c  -fPIC $(LDFLAGS) $(CFLAGS) $< -o $@ $(LIBS)


clean:
	rm -f  *.o *.so *.a
