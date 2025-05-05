message("[EXTERNAL]")

set(CDIR ${CMAKE_CURRENT_LIST_DIR})

set(EXTERNAL_DIRS
  ${CDIR}
  ${CDIR}/mmio/
)

include_directories(PUBLIC ${EXTERNAL_DIRS})

add_library( mmio SHARED IMPORTED )
set_target_properties( mmio PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/mmio/libmmio.so )

set(EXTERNAL_DIRS ${EXTERNAL_DIRS} PARENT_SCOPE)
