#define LEARNING_RATE 0.5
#define DISCOUNT 0.9
#define ITER_COUNT 50000
#define EXPLORE 3.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <gsl/gsl_rng.h>

unsigned long int task, state, action;
long int *trans = NULL;
double *reward = NULL, *q = NULL, explore, a = LEARNING_RATE, g = DISCOUNT;

gsl_rng *rng = NULL;

struct list_elem{
    //An element of a linked list where each link contains a character.
    char val;
    struct list_elem *next;
};

struct list{
    // Simple list element, note the size parameter
    struct list_elem *head;
    struct list_elem *tail;
    int size;
};

int prepare_input(char **, FILE *);
void mtql();
int has_reward(unsigned long int);
double max_q(unsigned long int, int);

struct list * li_alloc();
void li_free(struct list *li);
void li_elem_free(struct list_elem *current);
int li_add(struct list *li, char token);

int main(int argc, char **argv)
{
    if(argc < 2 || argc > 3){ 
        fprintf(stderr,"%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s%s%s",
            "Error:  No input file specified.",
            "The file should contain one line which is a space seperated",
            "stream of numbers. First should be three positive integers. Lets", "call them T, S and A representing the problems number of tasks,",
            "states and actions respectivily.",
            "Next should be an integer matrix with dimensions SxA written row",
            "by row, representing the problems transition matrix where -1 is",
            "an unavailable state, action pair.",
            "Lastly there should be double matrix with dimensions SxT written",
            "row by row, representing the problems rewards matrix.\n\t",
            "A last optional parameter is a non negative integer to use as a\n",
            "\tseed for random number generation.");
        return 1;
    }
    
    unsigned long int ticker;
    int i, j, t;
    long int seed;
    char *input = NULL;
    char *pos = NULL;
    FILE *fs = NULL;
    if(!(fs = fopen(argv[1], "r"))){
        fprintf(stderr, "Error: Could not open input file %s.", argv[1]);
        return 1;
    }
    if(argc == 3)
        seed = strtol(argv[2], NULL, 0);
    else seed = 3;
    
    if(prepare_input(&input, fs)){
        fprintf(stderr, "Error:  Failed to extract string input from file %s.",
            argv[1]);
        fclose(fs);
        return 3;
    }
    fclose(fs);
    fs = NULL;
    
    task = strtoul(input, &pos, 0);
    state = strtoul(pos, &pos, 0);
    action = strtoul(pos, &pos, 0);
    
    trans = (long int *)malloc(sizeof(long int)*state*action);
    reward = (double *)malloc(sizeof(double)*state*task);
    q = (double *)malloc(sizeof(double)*task*state*action);
    
    for(i = 0; i < state * action; ++i){
        trans[i] = strtol(pos, &pos, 0);
    }
    
    for(i = 0; i < state * task; ++i){
        reward[i] = strtod(pos, &pos);
    }
    
    free(input);
    input = NULL;
    pos = NULL;

    for(i = 0; i < task * state * action; ++i){
        q[i] = 0.0;
    }
    
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
    explore = EXPLORE;
    for(ticker = 0; ticker < ITER_COUNT; ++ticker){
        mtql();
    }
    if(rng) gsl_rng_free(rng);
    rng = NULL;
    
    fs = stdout; //TODO: Should be moved to output file name
    
    fprintf(fs, "transistion =\n[\n");
    for(i = 0; i < state; ++i){
        for(j = 0; j < action; ++j){
            fprintf(fs, "  %ld", trans[i*action + j]);
        }
        fprintf(fs, ";\n");
    }
    fprintf(fs, "]\n");
    
    fprintf(fs, "reward =\n[\n");
    for(i = 0; i < state; ++i){
        for(j = 0; j < task; ++j){
            fprintf(fs, "  %8.4f", reward[i*task + j]);
        }
        fprintf(fs, ";\n");
    }
    fprintf(fs, "]\n");
    
    fprintf(fs, "taskc = %lu\n", task);
    for(t = 0; t < task; ++t){
        fprintf(fs, "q%d =\n[\n",t);
        for(i = 0; i < state; ++i){
            for(j = 0; j < action; ++j){
                fprintf(fs, "  %8.4f", q[t*action*state + i*action + j]);
            }
            fprintf(fs, ";\n");
        }
        fprintf(fs, "]\n");
    }
    
    if(trans) free(trans);
    trans = NULL;
    if(reward) free(reward);
    reward = NULL;
    if(q) free(q);
    q = NULL;
    
    return 0;
}

int prepare_input(char **input, FILE * fs)
{
    int i;
    struct list *li;
    struct list_elem *temp;
    
    if(!(li = li_alloc())){
        fprintf(stderr, "Error: Failed to allocate list.");
        return 1;
    }
    
    while(!feof(fs)){
        li_add(li, fgetc(fs));
    }
    
    *input = (char *)malloc(sizeof(char)*(li->size + 1));
    
    temp = li->head;
    for(i = 0; i < li->size; ++i){
        (*input)[i] = temp->val;
        temp = temp->next;
    }
    (*input)[li->size] = '\0';
    
    return 0;
}

int has_reward(unsigned long int s)
{
    int i;
    
    for(i = 0; i < task; ++i){
        if(reward[s*task + i])
            return i+1;
    }
    
    return 0;
}

double max_q(unsigned long int s, int t)
{
    int i;
    double temp, best = q[t*state*action + s*action];
    for(i = 1; i < action; ++i){
        temp = q[t*state*action + s*action + i];
        if(best < temp)
            best = temp;
    }
    return best;
}

void mtql()
{
    unsigned long int pos, dest;
    unsigned int azi;
    int i, j;
    double temp, best;
    
    do pos = (unsigned long int)floor(gsl_rng_uniform(rng)*state);
        while(has_reward(pos));
    
    while(!has_reward(pos)){
        if(gsl_rng_uniform(rng) < explore){
            if(explore > 0.05) explore -= 0.01;
            do azi = (unsigned int)floor(gsl_rng_uniform(rng)*action);
                while(trans[pos*action + azi] == -1);
        } else {
            best = -99999;
            for(i = 0; i < task; ++i){
                for(j = 0; j < action; ++j){
                    temp = q[i*action*state + pos*action + j];
                    if(temp >= best){
                        best = temp;
                        azi = j;
                    }
                }
            }
            if(best == 0.0)
                do azi = (unsigned int)floor(gsl_rng_uniform(rng)*action);
                    while(trans[pos*action + azi] == -1);
        }
        dest = trans[pos*action + azi];
        for(i = 0; i < task; ++i){
            j = i*action*state + pos*action + azi;
            q[j] = q[j] + a*(reward[dest*task + i] + g*max_q(dest, i) - q[j]);
        }
        pos = dest;
    }
}

/* Returns a newly allocated list ready for use with
 *  a list function. Denoted by a "li_" suffix.
 */
struct list * li_alloc()
{
    struct list *li = NULL;
    
    if( (li = (struct list *) malloc(sizeof(struct list))) ){
        li->head = NULL;
        li->tail = NULL;
        li->size = 0;
    }
    else {
        fprintf(stderr, "Error: Could not allocate memory for a new list.\n");
    }
    
    return li;
}

/* Recursive function that clears the elements of a list starting from
 *  the given current element. 
 */
void li_elem_free(struct list_elem *current)
{
    if(current->next)
        li_elem_free(current->next);
    free(current);
}

/* Clears a given list along with all of it's elements. 
 */
void li_free(struct list *li)
{
    if(li){
        if(li->head){
            li_elem_free(li->head);
        }
        free(li);
    }
}

/* Adds the given character token to the specified list, returning 1 on
 *  a failure to allocate a new list element, otherwise the list size is
 *  increase and a 0 is returned. 
 */
int li_add(struct list *li, char token)
{
    struct list_elem *temp = NULL;
    
    if( !(temp = (struct list_elem *) malloc(sizeof(struct list_elem))) ){
        fprintf(stderr, "Error: Could not allocate memory for list element.\n");
        return 1;
    }
    
    temp->val = token;
    temp->next = NULL;
    
    if(li->head){
        li->tail->next = temp;
    } else {
        li->head = temp;
    }
    
    li->tail = temp;
    li->size++;
    
    return 0;
}

