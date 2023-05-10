import ctypes
 
#libObject = ctypes.CDLL('c:/Users/Julian/Documents/AP&AM22-23/Q3-Q4/Advanced Modeling/HarmonicsApproach/Attempt/function.so')

#libObject.test_empty()

#libObject.test_add(1, 2)

from function import *

print("Try test_empty:")
mylib.test_empty()

print("\nTry test_add:")
print(mylib.test_add(34.55, 23))

# Create a 25 elements array
numel = 25
data = (ctypes.c_int * numel)(*[x for x in range(numel)])
 
# Pass the above array and the array length to C:
print("\nTry passing an array of 25 integers to C:")
mylib.test_passing_array(data, numel)

print("data from Python after returning from C:")
for indx in range(numel):
    print(data[indx], end=" ")
print("")
