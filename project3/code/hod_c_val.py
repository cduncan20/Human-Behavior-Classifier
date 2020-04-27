def interface():
    print("HOD Hyperparameter 'C' Manager:")
    print("Choose the 'C' value for the HOD SVM training. 'C' must be greater than or equal to 0. Default value is 2.0.")
    c_val = choose_c_val()
    print("")
    return c_val


def initialize_with_default_values():
    print("Initializing with the default HOD 'C' value of 2.0.")
    c_val= 2.0
    print("")

    return c_val


def select_new_values():
    while True:
        c_val = float(input('C Value: '))
        while c_val < 0:
            print("Invalid 'C' value. Please enter a value for 'C' that is greater than or equal to 0.")
            c_val = float(input('C Value: '))

        print("You have selected to use C = {}.".format(c_val))
        confirmation = input("Is this correct? (y/n): ")
        while confirmation not in 'yn':
            confirmation = input("Invalid argument provided. Please select a valid option. Is the C value "
                                    "entered above correct?")

        if confirmation == 'y':
            break
        else:
            print("Please select new values")
            continue

    return c_val


def choose_c_val():
    print("Would you like to change the 'C' value from the default value?")
    selected_method = input("Type 'y' for yes or 'n' for no: ")
    while selected_method not in 'yn':
        print("Invalid argument provided. Please select a valid option. Would you like to change the 'C' value from"
              "from the default value?")
        selected_method = input("Type 'y' for yes or 'n' for no: ")

    if selected_method == 'y':
        c_val = select_new_values()
        return c_val
    else:
        print("Exiting HOD Hyperparameter 'C' Manager with default 'C' value.")
        c_val = initialize_with_default_values()
        return c_val
