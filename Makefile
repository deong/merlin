all : mtrlgen mtrlgenf mtqlproto mtqlseqgreed mtqlrandgreed mtqlbalgreed

% : %.c
	gcc -Wall -ggdb $< -o $@ -lgsl -lgslcblas -lm

cfg.txt.out : mtrlgenf cfg.txt
	./mtrlgenf cfg.txt

mtql.out : cfg.txt.out 
	./mtqlproto cfg.txt.out > mtql.m

.PHONY : clean
clean:
	for CFILE in *.c; do rm -f ${CFILE%%.c}; done
	rm -f cfg.txt.out mtql.out

