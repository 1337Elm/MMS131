"""
Python test file, to see if the files are correct CPT's of a Bayesian network or not

Author: Benjamin Elm Jonsson 2023, benjamin.elmjonsson@gmail.com
"""

def test_file(filename):
    """Function testing the CPT's of a given file. The conditions are the following. 
    The file has to have a length of 48, since this is the number of data points needed to fullfill the CPT (in this case).
    Each value needs to lie in the range [0,1].
    The sum of the probabilities for a child node given the parent node needs to be 1.
    """
    #Initiate variables needed later on
    correct_length = 48

    test_success = 0
    length = 0 
    correct_values = 0
    summa = 0

    data = {}

    #Read data file and put data into a dictionary
    with open(filename) as f:
        for row in f:
            length += 1

            (key,val) = row.split("=")

            if float(val) < 1 and float(val) > 0:
                correct_values += 1
            
            data[key] = float(val)
    f.close()
    
    #Define correct order
    order = ['a1','a2','a3','b1|a1','b2|a1','b3|a1','b1|a2','b2|a2','b3|a2','b1|a3','b2|a3','b3|a3','c1|a1',\
        'c2|a1','c3|a1','c1|a2','c2|a2','c3|a2','c1|a3','c2|a3','c3|a3','d1|a1,b1','d2|a1,b1','d3|a1,b1','d1|a1,b2',\
        'd2|a1,b2','d3|a1,b2','d1|a1,b3','d2|a1,b3','d3|a1,b3','d1|a2,b1','d2|a2,b1','d3|a2,b1','d1|a2,b2','d2|a2,b2','d3|a2,b2',\
        'd1|a2,b3','d2|a2,b3','d3|a2,b3','d1|a3,b1','d2|a3,b1','d3|a3,b1','d1|a3,b2','d2|a3,b2','d3|a3,b2','d1|a3,b3','d2|a3,b3','d3|a3,b3']
     
    #Test if file is OK (correct length, all values in [0,1] and every 3 elements should sum to 1)
    if length == correct_length:
        if correct_values == correct_length:
            counter = 0
            for i in order:
                if counter < 3:
                    summa += data[i]
                else:
                    counter = 0
                    if summa == 1:
                        test_success += 1
                        summa = data[i]
                    else:
                        print_fail(filename)
                        return
                counter += 1
            if test_success == 15:
                print_success(filename)
                return
        else:
            print_fail(filename)
            return
    else:
        print_fail(filename)
        return
    

def print_fail(filename):
    """Function that prints the message that the file has failed the tests and therefore
        does not determine the CPT
    """
    filename = filename.split('/Users/benjaminjonsson/Programmering/MMS131/1.3b/')[1]
    print(f"{filename} does not fully determine the CPT")


def print_success(filename):
    """Function that prints the message that the file has passed the tests and therefore
        does determine the CPT
    """
    filename = filename.split('/Users/benjaminjonsson/Programmering/MMS131/1.3b/')[1]
    print(f"{filename} fully determiens the CPT")


def main():
    """Main file that selects each file and sends it to test function
    """
    files = ['inNO1.txt','inNO2.txt','inNO3.txt','inOK1.txt','inOK2.txt']
    for i in files:
        filename = '/Users/benjaminjonsson/Programmering/MMS131/1.3b/' + i 
        test_file(filename)


if __name__ == '__main__':
    """Determines that the file is runnable, initiates the programme.
    """
    main()