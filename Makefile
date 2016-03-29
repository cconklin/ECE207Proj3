custom_functions.o: custom_functions.cu
	nvcc $^ -o $@ --optimize 3 -lm -lpthread -shared -Xcompiler -fPIC
