from add_student import add_student
from add_attendance import add_attendance
from train_model import retrain

print("="*72)
print("*****  ATTENDANCE SYSTEM USING DEEP LEARNING FOR FACE RECOGNITION  *****")

while True:
    print("="*72)
    print("What would you like to do?")
    print("[0] Exit")
    print("[1] Mark Attendance")
    print("[2] Add New Student")
    todo = input(">>> ")

    try:
        todo = int(todo)
        if todo == 0:
            print("Exiting...")
            break
        elif todo == 1:
            add_attendance()
            pass
        elif todo == 2:
            add_student()
            retrain()
            pass
        else:
            print("That's not a valid option!")
            continue
    except:
        print("That's not a valid option!")
