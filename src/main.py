from mining_data import make_dataset

print("1- Mining Data")
print("2- Start program")
keyboard_input = 1

if keyboard_input == 1:
    make_dataset()
elif keyboard_input == 2:
    print("Iniciando")
else:
    exit()
