import sys, getopt

def main(argv):
	flag = int(argv[0])
	s = -4 if flag > 5 else 1
	e = 1 if flag > 5 else 5
	reg_param_vals = [10 ** e for e in range(s,e)]
	print(reg_param_vals)

if __name__ == "__main__":
   main(sys.argv[1:])