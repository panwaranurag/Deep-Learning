import sys
def palindrome(input):
    length = len(input)
    for i in range(0,length/2):
        if input[i] == input[length-i-1]:
            continue
        else:
            sys.stdout.write("0")
            break
    sys.stdout.write("1")
input = raw_input()
palindrome(input)