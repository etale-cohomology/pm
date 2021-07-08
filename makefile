WATCH = pm.h makefile

# ----------------------------------------------------------------
all: pm  pm_bvh.cubin pm_pt.cubin

clean:
	rm -f  pm  pm_bvh.cubin pm_pt.cubin

# ----------------------------------------------------------------
pm_bvh.cubin: pm_bvh.cu  $(WATCH)
	t nvcc  $< -cubin -o $@  -cudart none -arch=sm_70 -use_fast_math -O3  -Xptxas "-O3 --verbose --warn-on-local-memory-usage --warn-on-spills"  -Xcudafe "--diag_suppress=declared_but_not_referenced"
pm_pt.cubin: pm_pt.cu  $(WATCH)
	t nvcc  $< -cubin -o $@  -cudart none -arch=sm_70 -use_fast_math -O3  -Xptxas "-O3 --verbose --warn-on-local-memory-usage --warn-on-spills"  -Xcudafe "--diag_suppress=declared_but_not_referenced"

# ----------------------------------------------------------------
pm: pm.c  $(WATCH)
	t gcc-8  $< -o $@  -lcuda -lX11 -lX11-xcb -lxcb -lGL  ${CFLAGS} ${CFAST}
