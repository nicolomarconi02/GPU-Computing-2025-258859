message("[EXTERNAL]")

set(CDIR ${CMAKE_CURRENT_LIST_DIR})

set(EXTERNAL_DIRS
  ${CDIR}
  ${CDIR}/mmio/
  ${CDIR}/expected
  ${CDIR}/expected/include/tl
)

include_directories(PUBLIC ${EXTERNAL_DIRS})

add_library(mmio
  ${CDIR}/mmio/mmio.c
)

set(EXTERNAL_DIRS ${EXTERNAL_DIRS} PARENT_SCOPE)
