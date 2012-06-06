mtrlgen : mtrlgen.c
	gcc -ggdb mtrlgen.c -o mtrlgen -lgsl -lgslcblas

.PHONY : clean
clean:
	rm -f mtrlgen
