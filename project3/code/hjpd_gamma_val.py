def interface():
    print("HJPD Hyperparameter 'gamma' Manager:")
    print("Choose the 'gamma' value for the HJPD SVM training. Gamma must be greater than or equal to 0. Default value is 0.125.")
    gamma = choose_gamma()
    print("")
    return gamma


def initialize_with_default_values():
    print("Initializing with the default HJPD of 0.125.")
    gamma= 0.125
    print("")

    return gamma


def select_new_values():
    while True:
        gamma = float(input('gamma value: '))
        while gamma < 0:
            print("Invalid gamma value. Please enter a value for gamma that is greater than or equal to 0.")
            gamma = float(input('gamma value: '))

        print("You have selected to use gamma = {}.".format(gamma))
        confirmation = input("Is this correct? (y/n): ")
        while confirmation not in 'yn':
            confirmation = input("Invalid argument provided. Please select a valid option. Is the gamma value "
                                    "entered above correct?")

        if confirmation == 'y':
            break
        else:
            print("Please select new values.")
            continue

    return gamma


def choose_gamma():
    print("Would you like to change the 'gamma' value from the default value?")
    selected_method = input("Type 'y' for yes or 'n' for no: ")
    while selected_method not in 'yn':
        print("Invalid argument provided. Please select a valid option. Would you like to change the 'gamma' value from"
              "from the default value?")
        selected_method = input("Type 'y' for yes or 'n' for no: ")

    if selected_method == 'y':
        gamma = select_new_values()
        return gamma
    else:
        print("Exiting HJPD Hyperparameter 'gamma' Manager with default 'gamma' value.")
        gamma = initialize_with_default_values()
        return gamma
