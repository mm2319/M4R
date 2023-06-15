

from two_compartment import create_data_twocompart

T,Y_1 = create_data_twocompart(p=0.25)
T,Y_2 = create_data_twocompart(p=0.25)
print(Y_1-Y_2)