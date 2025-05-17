# function to present simulation results as a table
def print_table_from_dict(mydict, n, n_d):
    print("----------------------------")
    print("Node\t Coin\t Probability")
    print("----------------------------")
    for count in mydict:
        print(count[0:n], '\t', count[n:n+n_d],'\t', mydict[count]/1000)
    print("----------------------------")
