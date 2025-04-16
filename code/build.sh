CXX=${CXX:-g++}
$CXX -std=c++17 -O2 -I. \
	-o ch_estimation_runner_2 \
	ch_estimation_runner_2.cpp \
	-lglog \
	-lxir \
	-lvart-runner \
	-lvitis_ai_library-graph_runner \


CXX=${CXX:-g++}
$CXX -std=c++17 -O2 -I. \
	-o ch_estimation_runner_3 \
	ch_estimation_runner_3.cpp \
	-lglog \
	-lxir \
	-lvart-runner \
	-lvitis_ai_library-graph_runner \