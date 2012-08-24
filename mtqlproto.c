#define LEARNING_RATE 0.5
#define DISCOUNT 0.9
#define ITER_COUNT 50000
#define EXPLORE 3.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <gsl/gsl_rng.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

unsigned long int task;
unsigned long int state;
unsigned long int action;
long int *trans = NULL;
double *reward = NULL;
double *q = NULL;
double explore;
double a = LEARNING_RATE;
double g = DISCOUNT;

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

/* Forward declarations */
int prepare_input(char **, FILE *);
void mtql();
int has_reward(unsigned long int);
double max_q(unsigned long int, int);

struct list * li_alloc();
void li_free(struct list *li);
void li_elem_free(struct list_elem *current);
int li_add(struct list *li, char token);

unsigned long get_seed();
/* End of forward declarations */

int main(int argc, char **argv)
{
    // Show error/help message when too few or too many arguments are provided.
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
    int i;
    int j;
    int t;
    unsigned long int seed;
    char *input = NULL;
    char *pos = NULL;
    FILE *fs = NULL;
    
    if(!(fs = fopen(argv[1], "r"))){
        fprintf(stderr, "Error: Could not open input file %s.\n", argv[1]);
        return 1;
    }
    
    // initiate seed as second argument or random if no second argument
    if(argc == 3)
        seed = strtoul(argv[2], NULL, 0);
    else seed = get_seed();
    
    // place the full contents of FILE fs into string input
    if(prepare_input(&input, fs)){
        fprintf(stderr, "Error:  Failed to extract string input from file %s.\n",
            argv[1]);
        fclose(fs);
        return 3;
    }
    
    fclose(fs);
    fs = NULL;
    
    // rip the first three values from input as task count, state count and action count respectivily
    task = strtoul(input, &pos, 0);
    state = strtoul(pos, &pos, 0);
    action = strtoul(pos, &pos, 0);
    
    // allocate matrices
    trans = (long int *)malloc(sizeof(long int)*state*action);
    reward = (double *)malloc(sizeof(double)*state*task);
    q = (double *)malloc(sizeof(double)*task*state*action);
    
    // rip the next state*action values as the transistion matrix
    for(i = 0; i < state * action; ++i){
        trans[i] = strtol(pos, &pos, 0);
    }
    
    // rip the next state*task values as the reward matrix
    for(i = 0; i < state * task; ++i){
        reward[i] = strtod(pos, &pos);
    }
    
    free(input);
    input = NULL;
    pos = NULL;

    // set the q values as zero
    for(i = 0; i < task * state * action; ++i){
        q[i] = 0.0;
    }
    
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
    explore = EXPLORE;
    
    // run the Q-learning algorithim ITER_COUNT times
    for(ticker = 0; ticker < ITER_COUNT; ++ticker){
        mtql();
    }
    
    if(rng) gsl_rng_free(rng);
    rng = NULL;
    
    fs = stdout; //TODO: Should be moved to output file name
    
    //print the transistion matrix, reward matrix, task count and Q-value matrices in a matlab style.
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

/* Prepares the input file as one string.
 *  Takes the given FILE pointer fs and saves every character
 *  one-by-one to a list(this is rather impractical). This list
 *  is then converted to a null terminated string *input.
 *  Returns 0: Success
 *          1: Failed to allocate list.
 *          2: Failed to allocate string.
 */
int prepare_input(char **input, FILE * fs)
{
    int i;
    struct list *li;
    struct list_elem *temp;
    
    if(!(li = li_alloc())){
        fprintf(stderr, "Error: Failed to allocate list.\n");
        return 1;
    }
    
    // get values and store in a list
    while(!feof(fs)){
        li_add(li, fgetc(fs));
    }
    
    if( (*input = (char *)malloc(sizeof(char)*(li->size + 1))) == NULL ){
        fprintf(stderr, "Error: Failed to allocate string.\n");
        li_free(li);
        return 2;
    }
    
    // copy list data to input
    temp = li->head;
    for(i = 0; i < li->size; ++i){
        (*input)[i] = temp->val;
        temp = temp->next;
    }
    
    // make input null terminated
    (*input)[li->size] = '\0';
    
    li_free(li);
    
    return 0;
}

/* Returns the first reward for state s.
 *  First is defined by the ordering of the tasks.
 */
int has_reward(unsigned long int s)
{
    int i;
    
    for(i = 0; i < task; ++i){
        if(reward[s*task + i])
            return i+1;
    }
    
    return 0;
}

/* Returns the best possible q-value for a given task t and state s.
 */
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

/* The Q-learning algoritim.
 *  Choooses a random non-reward starting state as the current state.
 *  Moves from state to state by choosing an action to take, the action is either
 *  chosen randomly when exploring(which happens less and less) or greedy.
 *  Each move updates the current stateXaction Q-values(for each state action pair
 *  it updates Q-values for each task) and then sets the destination state as the
 *  current state. Terminates when the current state is a reward state.
 */
void mtql()
{
    unsigned long int pos;
    unsigned long int dest;
    unsigned int azi;
    int i;
    int j;
    double temp;
    double best;
    
    // select starting state
    do pos = (unsigned long int)floor(gsl_rng_uniform(rng)*state);
        while(has_reward(pos));
    
    while(!has_reward(pos)){
        // run while current state has no reward
        if(gsl_rng_uniform(rng) < explore){
            // do an explore action
            if(explore > 0.05) explore -= 0.01; // decrease likelihood of explore, min 5% chance
            // select a viable action,
            // that is transistion graph value for the given state action pair is not -1
            do azi = (unsigned int)floor(gsl_rng_uniform(rng)*action);
                while(trans[pos*action + azi] == -1);
        } else {
            best = -99999; // Some -MAX_INT solution would be prefered here.
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
                // Q-values graph lacks knowledge of current state, therefore choose
                // a random viable action
                do azi = (unsigned int)floor(gsl_rng_uniform(rng)*action);
                    while(trans[pos*action + azi] == -1);
        }
        dest = trans[pos*action + azi];
        for(i = 0; i < task; ++i){
            // update the current state X chosen action Q-values on a per task basis
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

/* get a seed value for the random number generator
 *
 * tries to read bytes from /dev/random first; if that fails,
 * falls back to using the system clock.
 */
unsigned long get_seed()
{
    unsigned long seed;
    int fd = open("/dev/random", O_RDONLY|O_NONBLOCK);
    if((fd!=-1) && (read(fd, &seed, sizeof(unsigned long)) == sizeof(unsigned long))) {
        return (unsigned long)seed;
    } else {
        fprintf(stderr, "failed to read random bytes from /dev/random..."
                "falling back to system clock\n");
        return (unsigned long)(time(NULL));
    }
}
