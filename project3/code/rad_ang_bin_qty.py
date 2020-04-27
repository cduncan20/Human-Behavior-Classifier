def interface():
    print("RAD Angle Histogram Bin Manager:")
    print("Choose how many bins you would like to for the RAD Angle Histogram. Default value is 11 bins.")
    bin_qty = choose_bin_qty()
    print("")
    return bin_qty


def initialize_with_default_values():
    print("Initializing with the default RAD Angle bin quantity of 11 bins.")
    bin_qty = 11
    print("")

    return bin_qty


def select_new_values():
    while True:
        bin_qty = int(input('Bin Quantity: '))

        print("You have selected to use {} bins".format(bin_qty))
        confirmation = input("Is this correct? (y/n): ")
        while confirmation not in 'yn':
            confirmation = input("Invalid argument provided. Please select a valid option. Is the number "
                                    "of bins entered above correct?")

        if confirmation == 'y':
            break
        else:
            print("Please select new values")
            continue

    return bin_qty


def choose_bin_qty():
    print("Would you like to change the bin quantity from the default value?")
    selected_method = input("Type 'y' for yes or 'n' for no: ")
    while selected_method not in 'yn':
        print("Invalid argument provided. Please select a valid option. Would you like to change the bin quantity from"
              "from the default value?")
        selected_method = input("Type 'y' for yes or 'n' for no: ")

    if selected_method == 'y':
        bin_qty = select_new_values()
        return bin_qty
    else:
        print("Exiting RAD Angle Histogram Bin Manager with default bin value.")
        bin_qty = initialize_with_default_values()
        return bin_qty
