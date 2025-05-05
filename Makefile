CORES = $(shell nproc)
MAKEFLAGS += --no-print-directory

.PHONY: build clean

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
