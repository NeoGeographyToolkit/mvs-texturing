# TODO(oalexan1): Apparently rayint builds its own copy of mve, apart
# from the one below. This needs to be sorted out.

# TODO(oalexan1): Add all these as submodules rather than external projects,
# as then one can configure all of them at the same time.

# This is fetched and built as a submodule
# externalproject_add(ext_mapmap
#     PREFIX          ext_mapmap
#     GIT_REPOSITORY  git@github.com:NeoGeographyToolkit/mapmap_cpu.git
#     GIT_TAG         396f0fa1ce29f83de30e47c5b4fc7b2ca1e06e5d
#     UPDATE_COMMAND  ""
#     SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/mapmap
#     CONFIGURE_COMMAND ""
#     BUILD_COMMAND   ""
#     INSTALL_COMMAND ""
# )

# externalproject_add(ext_rayint
#     PREFIX          ext_rayint
#     GIT_REPOSITORY  git@github.com:NeoGeographyToolkit/rayint.git
#     GIT_TAG         cuda
#     UPDATE_COMMAND  ""
#     SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/rayint
#     CONFIGURE_COMMAND ""
#     BUILD_COMMAND   ""
#     INSTALL_COMMAND ""
# )

# Eigen is fetched separately.
#externalproject_add(ext_eigen
#    PREFIX          ext_eigen
#    URL             https://gitlab.com/libeigen/eigen/-/archive/3.3.2/eigen-3.3.2.tar.gz
#    URL_MD5         02edfeec591ae09848223d622700a10b
#    SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/eigen
#    CONFIGURE_COMMAND ""
#    BUILD_COMMAND   ""
#    INSTALL_COMMAND ""
#)

# externalproject_add(ext_mve
#     PREFIX          mve
#     GIT_REPOSITORY  git@github.com:NeoGeographyToolkit/mve.git
#     GIT_TAG         cfde01c
#     UPDATE_COMMAND  ""
#     SOURCE_DIR      ${CMAKE_SOURCE_DIR}/elibs/mve
#     CONFIGURE_COMMAND ""
#     BUILD_COMMAND   make -C libs/mve && make -C libs/util #not platform independent
#     BUILD_IN_SOURCE 1
#     INSTALL_COMMAND ""
# )
