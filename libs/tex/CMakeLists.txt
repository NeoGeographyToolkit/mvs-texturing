file (GLOB HEADERS "*.h")
file (GLOB SOURCES "[^_]*.cpp")

set(LIBRARY texture_reconstruction)
add_library(${LIBRARY} SHARED ${SOURCES})
#set_property(TARGET ${LIBRARY} PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
#add_dependencies(${LIBRARY} mve rayint mapmap)
target_link_libraries(${LIBRARY} TBB::tbb TBB::tbbmalloc mve
    ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${TIFF_LIBRARIES})

install(TARGETS ${LIBRARY} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
