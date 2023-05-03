try:
    from add_student import add_student
    from attendance import add_attendance, view_attendance
finally:
    pass

print("="*72)
print("*****  ATTENDANCE SYSTEM USING DEEP LEARNING FOR FACE RECOGNITION  *****")

while True:
    print("="*72)
    print("What would you like to do?")
    print("[0] Exit")
    print("[1] Mark Attendance")
    print("[2] Add New Student")
    print("[3] View Attendance")
    todo = input(">>> ")

    try:
        todo = int(todo)
        if todo == 0:
            print("Exiting...")
            print("=" * 72)
            break
        elif todo == 1:
            add_attendance()
            pass
        elif todo == 2:
            add_student()
            print("Added student... do not forget to run `train_model.py`")
        elif todo == 3:
            view_attendance()
        else:
            print("That's not a valid option!")
    except ValueError:
        print("That's not a valid option")
