all : mtrlgen mtrlgenf mtqlproto mtql.out

mtrlgen : mtrlgen.c
	gcc -Wall -ggdb mtrlgen.c -o mtrlgen -lgsl -lgslcblas -lm

mtrlgenf : mtrlgenf.c
	gcc -Wall -ggdb mtrlgenf.c -o mtrlgenf -lgsl -lgslcblas -lm

mtqlproto : mtqlproto.c
	gcc -Wall -ggdb mtqlproto.c -o mtqlproto -lgsl -lgslcblas -lm

cfg.txt.out : mtrlgenf cfg.txt
	./mtrlgenf cfg.txt

mtql.out : cfg.txt.out 
	./mtqlproto cfg.txt.out > mtql.out

.PHONY : clean
clean:
	rm -f mtrlgen mtrlgenf mtqlproto cfg.txt.out mtql.out

