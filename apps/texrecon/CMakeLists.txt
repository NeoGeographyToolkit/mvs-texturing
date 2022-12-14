file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

set(BIN texrecon_bin)

add_executable(${BIN} ${SOURCES})
#set_property(TARGET ${BIN} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
#add_dependencies(${BIN} ext_mve)
target_link_libraries(${BIN} texture_reconstruction TBB::tbb TBB::tbbmalloc mve mve_util
    ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES}
)
install(TARGETS ${BIN} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# TODO(oalexan1): Note sure about the robustness of the logic below but it seems to work
if(APPLE)
  set_target_properties(${BIN} PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../lib")
elseif(UNIX) # Unix which is not Apple
  set_target_properties(${BIN} PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:${MULTIVIEW_DEPS_DIR}/lib")
endif()
