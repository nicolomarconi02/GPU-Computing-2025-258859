CORES = $(shell nproc)
MAKEFLAGS += --no-print-directory

.PHONY: build clean remove-session remove-outputs

build:
	mkdir -p build
	@if [ -d bin ]; then \
		echo "Removing bin folder";\
		rm -rf bin;\
	fi;\
	cd build;\
	cmake ..;\
	make -j${CORES}

clean:
	@if [ -d build ]; then \
		echo "Removing build folder";\
		rm -rf build;\
	fi;

remove-session:
	@if [ -d profiler_session ]; then \
		echo "Removing profiler_session folder";\
		rm -rf profiler_session;\
	fi;

remove-outputs:
	@if [ -d output_matrices ]; then \
		echo "Removing output_matrices folder";\
		rm -rf output_matrices;\
	fi;
	@if [ -d output_run ]; then \
		echo "Removing output_run folder";\
		rm -rf output_run;\
	fi;
