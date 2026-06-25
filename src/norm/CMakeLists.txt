add_library(norm_lib reference.cpp layer_norm_ssve.S)
target_compile_options(norm_lib PRIVATE -march=armv9-a+sme)

add_executable(test_norm ../../tests/test_norm.cpp)
target_link_libraries(test_norm PRIVATE norm_lib Catch2::Catch2WithMain)

add_executable(main_norm ../../apps/main_norm.cpp)
target_link_libraries(main_norm PRIVATE norm_lib)

add_test(NAME NormTests COMMAND test_norm)
