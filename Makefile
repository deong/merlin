mtrlgen : mtrlgen.c
	gcc -Wall -ggdb mtrlgen.c -o mtrlgen -lgsl -lgslcblas

.PHONY : clean
clean:
	rm -f mtrlgen

