#define B_SIZE 4096
#define WHITE_SPACE " \n\r\t\v\f"
#define DIGIT "0123456789"
#define DELIMITER " ,;\n\t\v\f\r"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>

/* Struct definitions */

struct maze_struct {
    unsigned int height;
    unsigned int width;
    unsigned int goalcount;
    unsigned int start; // Start state of the maze.
    unsigned int *goal; // An array of states the size of goalcount,
                        // where each state is a goal.
    gsl_matrix *pathways; // A (height*width) X 4 matrix, or
                          // states X actions, which defines a pathway
                          // out of a given state in a given direction.
} *maze = NULL; // using global pointer maze as a pointer to the generated maze.

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

struct buffered_read{
    // This entity maintains the input file stream and reads a maximum of
    // B_SIZE characters from the stream to a buffer, the buffer is refilled
    // when needed.
    // Also maintains line which is the current line in the input file
    // for error handling. Line is not 100% reliable.
    FILE *fs;
    char *buffer;
    char *current;
    char *end;
    int offset;
    int line;
};

/* End of struct definitions */

/* Forward definitions */
void print_usage();
int  generate_instance(gsl_matrix** samples, int n, int m, int k, gsl_matrix* r, 
		       gsl_vector *mean, gsl_vector *stddev, double density, unsigned long seed,
               gsl_vector *lower, gsl_vector *upper);
unsigned long get_seed();
gsl_matrix* pearson_correlation(gsl_matrix* r);
void print_to_matlab(int n, int m, int k, gsl_matrix *rewards, char* name);

struct list * li_alloc();
void li_free(struct list *li);
void li_elem_free(struct list_elem *current);
int li_add(struct list *li, char token);
struct buffered_read * br_alloc(const char *filename);
void br_free(struct buffered_read *br);
char * br_get(struct buffered_read *br);
void br_jam(struct buffered_read *br, const char *until);
char * br_peek(struct buffered_read *br);
void maze_alloc(unsigned int h, unsigned int w, unsigned int tasks);
void maze_free();
int is_not_goal(int c);
char is_goal(int state);
void maze_random(unsigned long seed, unsigned int extra);
int prepare_maze_output(FILE *fs);
int br_cmp(struct buffered_read *br, const char *prestring, const char *match);
int contains(char *token, const char *check);
char * br_skip(struct buffered_read *br);
int br2int(struct buffered_read *br, const char *action_name);
double br2double(struct buffered_read *br, const char *action_name);
int get_class(struct buffered_read *br);
int get_str(struct buffered_read *br, char **output, const char *start,
            char *delimiter, const char *action_name);
gsl_vector * create_gsl_vector(char *input, int count);
gsl_matrix * create_gsl_matrix(char *input, int rows, int columns);
void print_gsl_vector(FILE *fs, gsl_vector *input, const char *comment,
            const char *name);
void print_gsl_matrix(FILE *fs, gsl_matrix *input, const char *comment,
            const char *name);
void print_maze(FILE *fs);
void print_data(FILE *fs, const char *comment, int class, int tasks, int states, int actions,
                int seed_given, double density, char *file_name, gsl_matrix *user_corr,
                gsl_vector *mean, gsl_vector *stddev, gsl_vector *lower, gsl_vector *upper);

gsl_rng *rng = NULL;

/* Should print a help message as instructions for the user.
 */
void print_usage()
{
    printf("TODO: Add help text. \n");
}

/* Returns a newly allocated list ready for use with
 *  a list function. List functions are denoted by a "li_" suffix.
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

/* Returns a pointer to a newly allocated buffered reader if memory
 *  allows, otherwise it returns a null pointer. 
 */
struct buffered_read * br_alloc(const char *filename)
{
    struct buffered_read *br = NULL;
    
    if( (br = (struct buffered_read *) malloc(sizeof(struct buffered_read))) ){
        if( (br->fs = fopen(filename, "r")) ){
            if( (br->buffer = (char*) malloc(sizeof(char*) * (B_SIZE + 1))) ){
                br->end = &br->buffer[B_SIZE];
                br->current = br->end;
                *br->end = '\0';
                br->offset = 0;
                br->line = 1; 
            }
            else {
                fprintf(stderr, 
                  "Error: Could not allocate memory buffer for reading of file %s.\n", filename);
                fclose(br->fs);
                free(br);
                br = NULL;
            }
        }
        else {
            fprintf(stderr, "Error: Could not open/create file %s.\n", filename);
            free(br);
            br = NULL;
        }
    }
    else{
        fprintf(stderr, "Error: Could not allocate memory.\n");
        br = NULL;
    }
    
    return br;
}
/* Deallocates the dynamic memory of specified buffered reader.
 */
void br_free(struct buffered_read *br)
{
    if(br){
        if(br->buffer) free(br->buffer);
        if(br->fs) fclose(br->fs);
        free(br);
    }
}

/* Returns a pointer to the current character and then increments the  
 *  current character to the next. Automatically flushes and refills
 *  the buffer if needed. Returns a null pointer if the file and
 *  buffer has been read in full. 
 */
char * br_get(struct buffered_read *br)
{
    char *val = br_peek(br);
    if(br->offset) br->current++;
    
    // Side effect:
    // Increments the line counter when pointing to vertical whitespace.
    if( val && contains(val, "\n\v\f")) br->line++;
    
    return val;
}

/* This function cycles through the given buffered reader br until
 *  the given end string is reached.
 */
void br_jam(struct buffered_read *br, const char *until)
{
    char *temp = NULL;
    while((temp = br_peek(br)) ){
        if(until && (until[0] == *temp) && !br_cmp(br, NULL, until) )
            break;
        br_get(br);
    }
}

/* If EOF and the buffer has been read then this returns a NULL pointer.
 *  Otherwise a pointer to the current character is returned
 *  without incrementing. 
 */
char * br_peek(struct buffered_read *br)
{
    if( feof(br->fs) && (br->current == br->end) ) return NULL;
    
    if( br->current == br->end ){
        br->offset = fread(br->buffer, 1, B_SIZE, br->fs);
        br->end = &(br->buffer[br->offset]);
        br->current = br->buffer;
    }
    
    if(br->current != br->end && *(br->current) == '/'){
        if(br->current != (br->end - 1) && *(br->current + 1) == '*'){
            br->current = br->current + 2;
            br_jam(br, "*/");
        }
        else if(br->current == (br->end - 1)){
            fpos_t pos;
            fgetpos(br->fs, &pos);
            char c = fgetc(br->fs);
            fsetpos(br->fs, &pos);
            if( c == '*'){
                br->current++;
                br_get(br);
                br_jam(br, "*/");
            }
        }
    }
    
    return (br->current);
}

/* Allocates memory for the global pointer maze and sets the default
 *  values. Creates an hXw maze with all possible walls(that is pathways
 *  is a zero-matrix. Determines a randomized start and goal states. 
 */
void maze_alloc(unsigned int h, unsigned int w, unsigned int tasks)
{
    if(maze){
        fprintf(stderr, "Warning: An attempt was made to allocate a maze when one already exists\n");
    } else {
        if( (maze = (struct maze_struct *) malloc(sizeof(struct maze_struct))) ){
            int i;
            maze->goalcount = tasks;
            maze->start = (int)floor(gsl_rng_uniform(rng)*h*w);;
            maze->goal = (unsigned int *) malloc(sizeof(int)*tasks);
            maze->height = h;
            maze->width = w;
            maze->pathways = gsl_matrix_calloc(h*w, 4);
            for(i = 0; i < tasks; ++i){
                (maze->goal)[i] = (unsigned int) floor(gsl_rng_uniform(rng)*h*w);
            }
        } else {
            fprintf(stderr, "Error: Failed to allocate a maze.\n");
        }
    }
}

/* Frees all allocated memory for the maze. 
 */
void maze_free()
{
    if(maze){
        gsl_matrix_free(maze->pathways);
        free(maze->goal);
        free(maze);
        maze = NULL;
    } else {
        fprintf(stderr, "Warning: Attempt was made to free maze null pointer.\n");
    }
}

/* Checks if the next string of characters on the buffer is equal to the
 *  string given as match. Returns 1 on failure, 0 otherwise. 
 *  The string prestring is what has already been matched, for example
 *  if both "seed" and "state" are strings that might be part of the
 *  input, then if 's' is on the buffer the buffer needs to be advanced,
 *  if 'e' is now on the buffer then match is "eed" not "seed" and "s"
 *  is the prestring. This method was chosen for error reporting.
 *  Note: br_cmp() is case insensitive.
 */
int br_cmp(struct buffered_read *br, const char *prestring, const char *match)
{
    int length = strlen(match);
    int i;
    
    char *match_p = (char *)malloc(sizeof(char)*(length + 1));
    char *read = (char*)malloc(sizeof(char) * (length + 1));
    char *val;
    for(i = 0; i < length; i++)
        match_p[i] = match[i];
        
    for(i = 0; i < length; i++){
        if( (val = br_get(br)) ){
            read[i] = *(val);
            if( read[i] > 64 && read[i] < 91 && read[i] == (match[i] - 32) )
                match_p[i] = read[i];
            else if(read[i] != match[i]) break;
        } else {
            i--;
            break;
        }
    }
    if(++i < length){
        read[i] = '\0';
        if(prestring)
            fprintf(stderr, 
                  "Error: While reading line %d. Expected \"%s%s\", found %s%s.\n",
                  br->line, prestring, match_p, prestring, read);
        else fprintf(stderr,
                  "Error: While reading line %d. Expected \"%s\", found %s.\n",
                  br->line, match_p, read);
        free(read);
        return 1;
    }
    
    free(match_p);
    free(read);
    return 0;
}

/* Returns false if string check doesn't contain the character pointed
 *  to by token. 
 */
int contains(char *token, const char *check)
{
    if(token == NULL) return 0;
    int length = strlen(check);
    int i = 0;
    
    while(i < length){
        if(*token == check[i]) return 1;
        i++;
    }
    
    return 0;
}

/* Automated whitespace skipping.
 *  Returns a pointer to the first non-whitespace character
 *  or NULL if EOF and an empty buffer. 
 */
char * br_skip(struct buffered_read *br)
{
    while(contains(br_peek(br), WHITE_SPACE))
        br_get(br);
    return br_peek(br);
}

/* Reads and returns the next integer on the buffer. Jams the buffer if
 *  the input is non-viable.
 */
int br2int(struct buffered_read *br, const char *action_name)
{
    if( !(br_skip(br)) ){
        fprintf(stderr, 
          "Error: In line %d. Input value for %s is missing.\n", br->line, action_name);
    }
    
    int val = 0;
    int multiplier = 1;
    char *temp;
    
    if( (temp = br_get(br)) ){
        if( *temp == '-' ){
            multiplier = -1;
            val = 0;
        }
        else if( contains(temp, DIGIT) ){
            multiplier = 1;
            val = *temp - 48;
        }
        else if( *temp != '+' ){
            br_jam(br, NULL);
            fprintf(stderr, 
             "Error: In line %d. Found %c, when trying to read in an integer value for %s.\n",
             br->line, *temp, action_name);
        }
    
        while(contains(br_peek(br), DIGIT))
            val = val*10 + *(br_get(br)) - 48;
    }
    
    return val*multiplier;
}

/* Reads and returns the next double on the buffer. Jams the buffer if
 *  the input is non-viable.
 */
double br2double(struct buffered_read *br, const char *action_name)
{
    if( !(br_skip(br)) ){
        fprintf(stderr, "Error: In line %d. Input value for %s is missing.\n", br->line, action_name);
    }
    
    double val = 0.0;
    double multiplier = 1.0;
    char *temp;
    
    if( (temp = br_get(br) ) ){
        if( *temp == '-' ){
            multiplier = -1.0;
            val = 0.0;
        }
        else if( contains(temp, DIGIT) )
            val = (double)( *temp - 48 );
        else if( *temp != '+' ){
            br_jam(br, NULL);
            fprintf(stderr, 
             "Error: In line %d. Found %c, when trying to read in an integer value for %s.\n", 
             br->line, *temp, action_name);
        }
        
        while(contains( br_peek(br), DIGIT))
            val = val*10.0 + (double)( *(br_get(br)) - 48 );
        
        if(contains(br_peek(br),".")){
            br_get(br);
            
            while(contains( br_peek(br), DIGIT)){
                val = val*10.0 + (double)( *(br_get(br)) - 48 );
                multiplier *= 0.1;
            }
        }
        
        if(contains(br_peek(br), "eE") ){
            br_get(br);
            
            int exp = br2int(br, action_name);
            if(exp < 0)
                while(exp++ < 0) multiplier *= 0.1;
            else 
                while(exp-- > 0) multiplier *= 10.0;
        }    
    }
    return val*multiplier;
}

/* Checks to see if the input state doesn't matches any goal states.
 */
int is_not_goal(int state)
{
    int i;
    for(i = 0; i < maze->goalcount; ++i)
        if((maze->goal)[i] == state)
            return 0;
    return 1;
}

/* Returns ' ' if input state is not a goal state, otherwise returns
 *  the task for which state is the goal state.
 */
char is_goal(int state)
{
    int i;
    for(i = 0; i < maze->goalcount; ++i)
        if((maze->goal)[i] == state)
            return 48+i;
    return 32;
}

/* Expects the global pointer maze of type struct maze pointer to be preallocated
 *  using maze_alloc(unsigned int, unsigned  int, unsigned  int).
 *  maze_random breaks pathways such that from any two distinct states A and B there
 *  exists at least one valid and direct list of stateXaction pairs that change the
 *  current state from A to B. Furthermore due to an inertia parameter, pathways
 *  tend to be straight. The parameter seed is currently unused. The extra parameter
 *  breaks down additional walls allowing for cycles in the maze.
 */
void maze_random(unsigned long seed, unsigned int extra)
{
    if(maze){
        int rand = (int)floor(gsl_rng_uniform(rng)*4);
            // rand is the random direction.
            // 0: Up, 1: Right, 2: Down, 3: Left
        int h = maze->height;
        int w = maze->width;
        int m = h*w;
        int current = maze->start;
        char path[m];
        char new[m];
        int i;
        int j;
        double sum;
        
        for(i = 0; i < m; ++i){
            path[i] = 1;
            new[i] = 1;
        }
        path[current] = 0;
        
        while(is_not_goal(current)){
            // Create a path from start to a goal,
            // this path is updated directly to the path variable
            if(gsl_rng_uniform(rng) < 0.34) 
                // inertia, only selects a new direction a third of the time.
                rand = (int)floor(gsl_rng_uniform(rng)*4);
            switch(rand){
            case 0:
                if(current >= w){
                    gsl_matrix_set(maze->pathways, current, 0, 1.0);
                    current = current - w;
                    gsl_matrix_set(maze->pathways, current, 2, 1.0);
                }
                break;
            case 1:
                if(current%w != (w-1)){
                    gsl_matrix_set(maze->pathways, current, 1, 1.0);
                    current = current + 1;
                    gsl_matrix_set(maze->pathways, current, 3, 1.0);
                }
                break;
            case 2:
                if(current < w*(h-1)){
                    gsl_matrix_set(maze->pathways, current, 2, 1.0);
                    current = current + w;
                    gsl_matrix_set(maze->pathways, current, 0, 1.0);
                }
                break;
            case 3:
                if(current%w != 0){
                    gsl_matrix_set(maze->pathways, current, 3, 1.0);
                    current = current - 1;
                    gsl_matrix_set(maze->pathways, current, 1, 1.0);
                }
                break;
            }
            path[current] = 0;
        }
        
        for(i = 0; i < m; ++i){
            if(path[i]){
            // For each state i not in the path variable(that is path[i] = 1) create
            // a path new from i to state j such that for all states s in new, path[s] = 0
            // if and only if s = j.
                sum = 0.0;
                for(j = 0; j < 4; ++j){
                    sum += gsl_matrix_get(maze->pathways, i, j);
                }
                if(sum == 0.0){
                    current = i;
                    rand = (int)floor(gsl_rng_uniform(rng)*4);
                    new[current] = 0;
                    while(path[current]){
                        if(gsl_rng_uniform(rng) < 0.34)
                            rand = (int)floor(gsl_rng_uniform(rng)*4);
                        switch(rand){
                        case 0:
                            if(current >= w){
                                gsl_matrix_set(maze->pathways, current, 0, 1.0);
                                current = current - w;
                                gsl_matrix_set(maze->pathways, current, 2, 1.0);
                            }
                            break;
                        case 1:
                            if(current%w != (w-1)){
                                gsl_matrix_set(maze->pathways, current, 1, 1.0);
                                current = current + 1;
                                gsl_matrix_set(maze->pathways, current, 3, 1.0);
                            }
                            break;
                        case 2:
                            if(current < w*(h-1)){
                                gsl_matrix_set(maze->pathways, current, 2, 1.0);
                                current = current + w;
                                gsl_matrix_set(maze->pathways, current, 0, 1.0);
                            }
                            break;
                        case 3:
                            if(current%w != 0){
                                gsl_matrix_set(maze->pathways, current, 3, 1.0);
                                current = current - 1;
                                gsl_matrix_set(maze->pathways, current, 1, 1.0);
                            }
                            break;
                        }
                        new[current] = 0;
                    }
                    for(j = 0; j < m; ++j){
                    // Update the path variable so it contains the new path in new and reset new.
                        path[j] *= new[j];
                        new[j] = 1;
                    }
                }
            }
        }
        
        //And break a few extra pathways
        while(extra){
            current = (int)floor(gsl_rng_uniform(rng)*m);
            rand = (int)floor(gsl_rng_uniform(rng)*4);
            switch(rand){
            case 0:
                if(current >= w &&
                   gsl_matrix_get(maze->pathways, current, rand) == 0){
                    extra--;
                    gsl_matrix_set(maze->pathways, current, rand, 1);
                }
                break;
            case 1:
                if(current%w != (w-1) &&
                   gsl_matrix_get(maze->pathways, current, rand) == 0){
                    extra--;
                    gsl_matrix_set(maze->pathways, current, rand, 1);
                }
                break;
            case 2:
                if(current < w*(h-1) &&
                   gsl_matrix_get(maze->pathways, current, rand) == 0){
                    extra--;
                    gsl_matrix_set(maze->pathways, current, rand, 1);
                }
                break;
            case 3:
                if(current%w != 0 &&
                   gsl_matrix_get(maze->pathways, current, rand) == 0){
                    extra--;
                    gsl_matrix_set(maze->pathways, current, rand, 1);
                }
                break;
            }
        }
        
    } else {
        fprintf(stderr, "Warning: Attempt to randomize maze null pointer.\n");
    }
}

/* Returns a simple integer to signify the class of the current problem.
 *  Currently only recognizes "maze" as an input. Returns -1 on failure.
 */
int get_class(struct buffered_read *br)
{
    int class;
    char *temp = br_skip(br);
    if(temp == NULL){
        fprintf(stderr, "Error: End of file reached before promised class definition.\n");
        return -1;
    }
    
    switch(*temp){
    case 'm':
    case 'M':
        class = br_cmp(br, NULL, "maze") ? -1 : 1; break;
    case 'r':
    case 'R':
        class = br_cmp(br, NULL, "random problem") ? -1 : 2; break;
    
    default:
        fprintf(stderr, "Error: No class starts with %c\n", *temp);
        class = -1;
    }
    
    return class;
}

/* Returns the string starting from the current position of the buffered reader
 *  until one of the delimiter characters is found. If no delimter characters are
 *  specified then "\n" is used instead. If start is not NULL then the search is
 *  begins after start is read and will only find numbers.
 *  Returns 0 when successful.
 */
int get_str(struct buffered_read *br, char **output, const char *start,
            char *delimiter, const char *action_name)
{
    if( !(br_skip(br)) ){
        fprintf(stderr, 
          "Error: In line %d. Input value for %s is missing.\n", 
          br->line, action_name);
    }
    
    if( start && !(contains(br_get(br), start)) ){
        fprintf(stderr,
          "Error: in line %d. Input value for %s should start one of the following tokens: %s.\n", 
          br->line, action_name, start);
        return 1;
    }
    
    struct list *li = NULL;
    struct list_elem *e = NULL;
    char *temp;
    int index = 0;
    
    if( !(li = li_alloc()) ){
        fprintf(stderr,
            "Error: In line %d. Could not allocate memory to complete %s", br->line, action_name);
        return 2;
    }
    
    if(delimiter == NULL || strlen(delimiter) == 0)
        delimiter = "\n";
        
    if(start)
        // only numeric values, returns a list of space seperated numbers
        while( (temp = br_peek(br)) && !(contains(temp, delimiter)) ){
            br_get(br);
            if( contains(temp, DIGIT) || contains(temp, "-+eE") ){
                li_add(li, *temp);
            } else if( *temp == '.' && contains( br_peek(br), DIGIT ) ) {
                li_add(li, '.');
            } else if( (li->tail) && contains(temp, WHITE_SPACE) && li->tail->val != ' ' ) {
                li_add(li, ' ');
            }
            else{
                fprintf(stderr,
                    "Error: In line %d. Found %c when trying to fill %s.",
                    br->line, *temp, action_name);
                return 3;
            }
        }
    else
        // anything up to the next delimiter token
        while( (temp = br_peek(br)) && !(contains(temp, delimiter)) ){
            br_get(br);
            li_add(li, *temp);
        }
    
    if( temp && !(contains(temp, delimiter)) ){
        fprintf(stderr, 
            "Error: In line %d. Did not find any of the expected delimiter tokens( %s ) for %s", 
            br->line, delimiter, action_name);
        free(li);
        return 3;
    } else {
        br_get(br);
    }
    
    *output = malloc(sizeof(char*) * li->size);
    e = li->head;
    
    while(e){
        (*output)[index++] = e->val;
        e = e->next;
    }
    
    (*output)[index] = '\0';
    
    li_free(li);
    
    return 0;
}

/* Returns a null pointer on failure, otherwise returns a pointer to a 
 *  newly allocated gsl_vector according to the string input created by
 *  a get_str() call with length equal to count.
 */
gsl_vector * create_gsl_vector(char *input, int count)
{
    int i;
    char *pos = input;
    gsl_vector *val = gsl_vector_alloc(count);
    
    for(i = 0; i < count; ++i){
        gsl_vector_set(val, i, strtod(pos, &pos));
    }
    
    return val;
}

/* Returns a null pointer on failure, otherwise returns a pointer to a 
 *  newly allocated rowsXcolumns gsl_matrix according to the string
 *  input created by a get_str() call.
 */
gsl_matrix * create_gsl_matrix(char *input, int rows, int columns)
{
    int i;
    int j;
    char *pos = input;
    gsl_matrix *val = gsl_matrix_alloc(rows, columns);
    
    for(i = 0; i < rows; ++i){
        for(j = 0; j < columns; ++j){
            gsl_matrix_set(val, i, j, strtod(pos, &pos));
        }
    }
    
    return val;
}

/* Prints the global pointer maze to the file stream fs.
 */
void print_maze(FILE *fs)
{
    int i, j;
    int w = maze->width;
    int h = maze->height;
    
    fprintf(fs, "\n");
    for(i = 0; i < h; ++i){
        for(j = 0; j < w; ++j){
            fprintf(fs, "#");
            fprintf(fs, "%c", gsl_matrix_get(maze->pathways, i*w+j, 0) ? '.' : '#');
        }
        fprintf(fs, "#\n#");
        for(j = 0; j < w; ++j){
            if(i*w+j == maze->start)
                fprintf(fs, "S");
            else fprintf(fs, "%c", is_goal(i*w+j));
            fprintf(fs, "%c", gsl_matrix_get(maze->pathways, i*w+j, 1) ? '.' : '#');
        }
        fprintf(fs, "\n");
    }
    for(i = 0; i < w; ++i)
    {
        fprintf(fs, "##");
    }
    fprintf(fs, "#\n\n");
}

/* Creates and prints to file stream fs the specification of the maze.
 *  Specifications are number of tasks, states and actions along with
 *  a transistion matrix and rewards matrix.
 */
int prepare_maze_output(FILE *fs)
{
    if(maze){
        //  task, states, actions
        //  state transition matrix: (state x action) -1 for wall, number of new state otherwise [spelling]
        //  reward matrix: ((state*action) x task) 0 except into goal state for applicable task
        
        unsigned int t = maze->goalcount;
        unsigned int w = maze->width;
        unsigned int s = maze->height * w;
        unsigned int a = 4;
        unsigned int r = s*a*10;
        gsl_matrix *m = maze->pathways;
        gsl_matrix *transition = gsl_matrix_alloc(s, a);
        gsl_matrix *rewards = gsl_matrix_calloc(s, t);
        //gsl_matrix *rewards = gsl_matrix_calloc(s*a, t);
        int i,j,d;
        
        for(i = 0; i < s; ++i){
            d = gsl_matrix_get(m, i, 0) ? i-w : -1;
            gsl_matrix_set(transition, i, 0, d);
            
            d = gsl_matrix_get(m, i, 1) ? i+1 : -1;
            gsl_matrix_set(transition, i, 1, d);
            
            d = gsl_matrix_get(m, i, 2) ? i+w : -1;
            gsl_matrix_set(transition, i, 2, d);
            
            d = gsl_matrix_get(m, i, 3) ? i-1 : -1;
            gsl_matrix_set(transition, i, 3, d);
        }
        
        for(i = 0; i < t; ++i){
            gsl_matrix_set(rewards, (maze->goal)[i], i, r);
        }
        
        fprintf(fs, "%d %d %d ", t, s, a);
        for(i = 0; i < s; ++i) for(j = 0; j < a; ++j)
            fprintf(fs, "%d ", (int)gsl_matrix_get(transition, i, j));
        for(i = 0; i < s; ++i) for(j = 0; j < t; ++j)
            fprintf(fs, "%8.4f ", gsl_matrix_get(rewards, i, j));
        
    } else {
        fprintf(stderr, "Error: Attempt made to prepare maze output when maze unallocated.");
        return 1;
    }
    
    return 0;
}

/* Prints the gsl_matrix input to the filestream fs with the title name and each line prefaced
 *  with the given comment string.
 */
void print_gsl_vector(FILE *fs, gsl_vector *input, const char *comment, const char *name)
{
    if(input){
        int count = input->size;
        int i;
        
        fprintf(fs, "\n%s%s =\n%s[\n%s",
            comment, name ? name : "unnamed vector", comment, comment);
        
        for(i = 0; i < count; ++i){
            fprintf(fs, "%8.4f  ", gsl_vector_get(input, i));
        }
        
        fprintf(fs, "\n%s]\n", comment);
            
    } else
        fprintf(fs, "%sThe %s has yet to be allocated.\n",
            comment, name ? name : "unnamed vector");
}

/* Prints the gsl_matrix input to the filestream fs with the title name and each line prefaced
 *  with the given comment string.
 */
void print_gsl_matrix(FILE *fs, gsl_matrix *input, const char *comment, const char *name)
{
    if(input){
        int rows = input->size1;
        int columns = input->size2;
        int i;
        int j;
        
        fprintf(fs, "\n%s%s =\n%s[\n", comment, name ? name : "unnamed matrix", comment);
        
        for(i = 0; i < rows; ++i){
            fprintf(fs, "%s", comment);
            for(j = 0; j < columns; ++j){
                fprintf(fs, "%8.4f  ", gsl_matrix_get(input, i, j));
            }
            fprintf(fs, "\n");
        }
        
        fprintf(fs, "%s]\n", comment);
    } else
        fprintf(fs, "%sThe %s has yet to be allocated.\n",
            comment, name ? name : "unnamed matrix");
}

/* print_data() prints all the data to the given filestream, prefaced by the given comment string.
 */
void print_data(FILE *fs, const char *comment, int class, int tasks, int states, int actions,
                int seed_given, double density, char *file_name, gsl_matrix *user_corr,
                gsl_vector *mean, gsl_vector *stddev, gsl_vector *lower, gsl_vector *upper)
{
    int i;
    char *comment_s = (char *)malloc(sizeof(char)*(strlen(comment) + 3));
    
    for(i = 0; i < strlen(comment); i++)
        comment_s[i] = comment[i];
    comment_s[i] = ' ';
    comment_s[i+1] = ' ';
    comment_s[i+2] = '\0';
    
    fprintf(fs, "\n%sclass: %d\n%s#task: %d, #state: %d, #action: %d\n",
            comment_s, class, comment_s, tasks, states, actions);
    fprintf(fs, "%sOutput to file: %s, density: %8.2f, seed: %d\n",
            comment_s, file_name, density, seed_given);
    
    print_gsl_matrix(fs, user_corr, comment_s, "Correlation matrix");
    
    print_gsl_vector(fs, stddev, comment_s, "Standard deviation vector");
    
    print_gsl_vector(fs, mean, comment_s, "Mean vector");
    
    print_gsl_vector(fs, lower, comment_s, "Lower bounds vector");
    
    print_gsl_vector(fs, upper, comment_s, "Upper bounds vector");
    
    free(comment_s);
}

int main(int argc, char **argv)
{
    if(argc != 2){
        fprintf(stderr, "Currently only accepting one argument which should be the name of a config file in the same directory or \"HELP\"\n");
        return 1;
    }
    
    if(strcmp(argv[1],"HELP") == 0 || strcmp(argv[1],"help") == 0 || strcmp(argv[1],"Help") == 0){
        print_usage();
        return 0;
    }

    struct buffered_read *br = NULL;
    FILE *fs = NULL;
    
    unsigned int class = 0;
    unsigned int states = 0;
    unsigned int actions = 0;
    unsigned int tasks = 0;
    unsigned long seed_given = get_seed();
    double density = 0.1;
    char *file_name = NULL;
    gsl_matrix *user_corr = NULL;
    char *corr_str = NULL;
    gsl_matrix *calc_corr = NULL;
    gsl_matrix *rewards = NULL;
    gsl_vector *mean = NULL;
    char *mean_str = NULL;
    gsl_vector *stddev = NULL;
    char *stddev_str = NULL;
    gsl_vector *lower = NULL;
    char *lower_str = NULL;
    gsl_vector *upper = NULL;
    char *upper_str = NULL;
    char *comment = NULL;
    
    char *read = NULL;
    char *temp;
    int i;
    
    br = br_alloc(argv[1]);
    
    while( (temp = br_peek(br)) ){
    // Loop will run until the buffered reader has seen the whole file.
    read = "\0\0\0\0";  // read is used to store read characters shared by more than one keyword,
                        // after seeing 'c' then read = "c\0\0\0" for keywords "class" and "correlation".
                        // WARNING: If a keyword is added such that two(or more) keywords share more than
                        // the first three letters, then the length of read will need to be increased accordingly.
        switch(*temp){
        // The current character is used to match to a keyword where possible
        // else to a group of keywords all starting on the current character.
        case 'a':   // Number of actions
        case 'A':
            if( br_cmp(br, NULL, "actions:") ){
                br_free(br);
                return 1;
            }
            actions = br2int(br, "actions");
            break;
        case 'c':   // Class, Comment or Correlation
        case 'C':
            read[0] = *(br_get(br));
            if( (temp = br_peek(br)) ){
                if( *temp == 'l' || *temp == 'L'){ // Class
                    if( br_cmp(br, read, "lass:") ){
                        br_free(br);
                        return 1;
                    }
                    if( (class = get_class(br)) == -1){
                        br_free(br);
                        return 1;
                    }
                } else if( *temp == 'o' || *temp == 'O'){  // Comment or Correlation
                    read[1] = *(br_get(br));
                    if( (temp = br_peek(br)) ){
                        if( *temp == 'm' || *temp == 'M' ){ // Comment
                            if( br_cmp(br, read, "mment:") ){
                                br_free(br);
                                return 1;
                            }
                            if( get_str(br, &comment , NULL, WHITE_SPACE, "comment") ){
                                br_free(br);
                                return 1;
                            }
                        } else if( *temp == 'r' || *temp == 'R' ) { // Correlation
                            if( br_cmp(br, read, "rrelation:") ){
                                br_free(br);
                                return 1;
                            }
                            if( get_str(br, &corr_str , "[", "]", "rewards") ){
                                br_free(br);
                                return 1;
                            }
                        } else {
                            fprintf(stderr, 
                              "Error: While reading line %d. Unknown keyword starting with c%c.",
                              br->line, *temp);
                            br_free(br);
                            return 1;
                        }
                    } else {
                        fprintf(stderr,
                            "Error: While reading line %d. Encountered end of file after reading \"co\".", br->line);
                        br_free(br);
                        return 1;
                    }
                } else {
                    fprintf(stderr, 
                      "Error: While reading line %d. Unknown keyword starting with c%c.",
                      br->line, *temp);
                    br_free(br);
                    return 1;
                }
            } else {
                fprintf(stderr,
                  "Error: While reading line %d. Encountered end of file after reading \"c\".",
                  br->line);
            }
            break;
        case 'd':   // Density
        case 'D':
            if( br_cmp(br, NULL, "density:") ){
                br_free(br);
                return 1;
            }
            density = br2double(br, "density");
            break;
        case 'l':   // Array of lower Bounds, per task basis
        case 'L':
            if( br_cmp(br, NULL, "lower bound:") ){
                br_free(br);
                return 1;
            }
            if( get_str(br, &lower_str , "[", "]", "lower bounds") ){
                br_free(br);
                return 1;
            }
            break;
        case 'm':   // Array of mean values, per task basis
        case 'M':
            if( br_cmp(br, NULL, "mean:") ){
                br_free(br);
                return 1;
            }
            if( get_str(br, &mean_str , "[", "]", "mean") ){
                br_free(br);
                return 1;
            }
            break;
        case 'o':   // Output filename
        case 'O':
            if( br_cmp(br, NULL, "output filename:") ){
                br_free(br);
                return 1;
            }
            if( get_str(br, &file_name, NULL, DELIMITER, "output filename") ){
                br_free(br);
                return 1;
            }
            break;
        case 's':   // Seed (given), Standard deviation or number of states
        case 'S':
            read[0] = *(br_get(br));
            if( (temp = br_peek(br)) ){
                if(*temp == 'e' || *temp == 'E'){   // Seed
                    if( br_cmp(br, read, "eed:") ){
                        br_free(br);
                        return 1;
                    }
                    seed_given = (long) br2int(br, "seed");
                } else if (*temp == 't' || *temp == 'T'){
                    read[1] = *(br_get(br));
                    if( (temp = br_peek(br)) ){
                        if(*temp == 'a' || *temp == 'A'){
                            read[2] = *(br_get(br));
                            if( (temp = br_peek(br)) ){
                                if(*temp == 'n' || *temp == 'N'){   // Standard deviation
                                    if( br_cmp(br, read, "ndard deviation:") ){
                                        br_free(br);
                                        return 1;
                                    }
                                    if( get_str(br, &stddev_str , "[", "]", "standard deviation") ){
                                        br_free(br);
                                        return 1;
                                    }
                                } else if(*temp == 't' || *temp == 'T'){    // States
                                    if( br_cmp(br, read, "tes:") ){
                                        br_free(br);
                                        return 1;
                                    }
                                    states = br2int(br, "states");
                                } else {
                                    fprintf(stderr, 
                                      "Error: In line %d. Unknown keyword starting with sta%c.", br->line, *temp);
                                    br_free(br);
                                    return 1;
                                }
                            } else {
                                fprintf(stderr,
                                  "Error: While reading line %d. Encountered end of file after reading \"sta\".",
                                  br->line);
                            }
                        } else {
                            fprintf(stderr, 
                              "Error: In line %d. Unknown keyword starting with st%c.", br->line, *temp);
                            br_free(br);
                            return 1;
                        }
                    } else {
                        fprintf(stderr,
                          "Error: While reading line %d. Encountered end of file after reading \"st\".",
                          br->line);
                    }
                } else {
                    fprintf(stderr, 
                      "Error: In line %d. Unknown keyword starting with s%c.", br->line, *temp);
                    br_free(br);
                    return 1;
                }
            } else {
                fprintf(stderr,
                  "Error: While reading line %d. Encountered end of file after reading \"s\".",
                  br->line);
            }
            break;
        case 't':   // Number of tasks
        case 'T':
            if( br_cmp(br, NULL, "tasks:") ){
                br_free(br);
                return 1;
            }
            tasks = br2int(br, "tasks");
            break;
        case 'u':   // Array of upper bounds, per task basis
        case 'U':
            if( br_cmp(br, NULL, "upper bound:") ){
                br_free(br);
                return 1;
            }
            if( get_str(br, &upper_str, "[", "]", "upper bounds") ){
                br_free(br);
                return 1;
            }
            break;
        default:    // skip white space
            if(contains(temp, WHITE_SPACE))
                br_get(br);
            else {
                fprintf(stderr, "Error: In line %d. Found '%c' while searching for start of keyword.\n", br->line, *temp);
                return 1;
            }
        }
    }
    
    br_free(br);
    
    if(tasks < 1 || actions < 1){
        fprintf(stderr, "Error: The number of tasks and states must be defined as a positive integer.\n");
        return 1;
    }
    
    /* Set values from stored strings, or default where possible and needed. */
    if(corr_str) user_corr = create_gsl_matrix(corr_str, tasks, tasks);
    else {
        user_corr = gsl_matrix_alloc(tasks, tasks);
        gsl_matrix_set_identity(user_corr);
    }
    
    if(mean_str) mean = create_gsl_vector(mean_str, tasks);
    else {
        mean = gsl_vector_alloc(tasks);
        for(i = 0; i < tasks; ++i) gsl_vector_set(mean, i, 1.0);
    }
    
    if(stddev_str) stddev = create_gsl_vector(stddev_str, tasks);
    else {
        stddev = gsl_vector_calloc(tasks);
    }
    
    if(lower_str) lower = create_gsl_vector(lower_str, tasks);
    
    if(upper_str) upper = create_gsl_vector(upper_str, tasks);
    
    if(!file_name){
        // Create default output filename.
        int len = strlen(argv[1]);
        file_name = (char *) malloc(sizeof(char)*(len+5));
        for(i = 0; i < len; ++i) file_name[i] = argv[1][i];
        file_name[len] = '.';
        file_name[len+1] = 'o';
        file_name[len+2] = 'u';
        file_name[len+3] = 't';
        file_name[len+4] = '\0';
    }
    
    if(!(fs = fopen(file_name, "w"))){
        fprintf(stderr, "Error: Failed to open file %s.\n", file_name);
        return 1;
    }
    
    if(comment == NULL) comment = "";
    
    print_data(stdout, comment, class, tasks, states, actions, seed_given, density, file_name, user_corr, mean, stddev, lower, upper);
    
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed_given);
    
    switch(class){
    case 1: //maze
        maze_alloc(states, actions, tasks);
        maze_random(seed_given, states);
        print_maze(stdout);
        prepare_maze_output(fs);
        maze_free();
        break;
    case 2:
        if( generate_instance(&rewards, states, actions, tasks, user_corr,
            mean, stddev, density, seed_given, lower, upper) ){
              fprintf(stderr, "Error: Generating a random problem failed.\n");
              return 1;
        }
        print_to_matlab(states, actions, tasks, rewards, file_name);
        break;
    }
    
    gsl_rng_free(rng);
    
    /*gsl_matrix *calc_corr = NULL;
    gsl_matrix *rewards = NULL;*/ //These two are for calculated correlation and for actual rewards.
    
    if(file_name){
        free(file_name);
        if(fs) fclose(fs);
    }
    if(user_corr) gsl_matrix_free(user_corr);
    if(corr_str) free(corr_str);
    if(calc_corr) gsl_matrix_free(calc_corr);
    if(rewards) gsl_matrix_free(rewards);
    if(mean) gsl_vector_free(mean);
    if(mean_str) free(mean_str);
    if(stddev) gsl_vector_free(stddev);
    if(stddev_str) free(stddev_str);
    if(lower) gsl_vector_free(lower);
    if(lower_str) free(lower_str);
    if(upper) gsl_vector_free(upper);
    if(upper_str) free(upper_str);
    if(comment) free(comment);
    
    return 0;
}

/* Functions below are borrowed and retailored from mtrlgen.c */

int generate_instance(gsl_matrix** samples, int n, int m, int k, gsl_matrix* r, 
		      gsl_vector *mean, gsl_vector *stddev, double density, unsigned long seed,
              gsl_vector *lower, gsl_vector *upper) 
{
    /* the total number of samples for each task is nm (states * actions).
     * we'll reshape this into a matrix form when we're done. */
    const int N = m*n;
    gsl_matrix* v;
    gsl_matrix* T;
    gsl_matrix* randn;
    gsl_matrix *unmapped_samples;
    int i;
    int j;
    double p;
    const int nonzero_N = (int)ceil(N*density);
    //const int zero_N = N - nonzero_N;
    
    /* Memory allocation */
    unsigned int mapping_array[nonzero_N];
    //if(zero_N > 0)unsigned int zeros[zero_N];
    unsigned int zeros[N];
    T = gsl_matrix_alloc(r->size1, r->size2);
    v = gsl_matrix_alloc(nonzero_N, k);
    randn = gsl_matrix_calloc(nonzero_N, k);
    unmapped_samples = gsl_matrix_calloc(nonzero_N, k);
    *samples = gsl_matrix_calloc(N, k);
    
    /* take cholesky decomposition and make sure covariance matrix is positive-definite */
    gsl_matrix_memcpy(T, r);
    if(gsl_linalg_cholesky_decomp(T) == GSL_EDOM) {
        return 1;
    }
    
    /* make sure that mu is a column vector and copy it N times */
    //v = gsl_matrix_alloc(N, k);
    for(i=0; i<nonzero_N; ++i) {
        gsl_matrix_set_row(v, i, mean);
    }
    
    /* Original random generated matrix, will cause errors if used */
    /* create an Nxk matrix of normally distributed random numbers */
    /*rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
    randn = gsl_matrix_calloc(N, k);
    for(i=0; i<N; ++i) {
        for(j=0; j<k; ++j) {
            if (gsl_rng_uniform(rng) <= density)
                gsl_matrix_set(randn, i, j, gsl_ran_gaussian(rng, gsl_vector_get(stddev, j)));
            else
                gsl_matrix_set(v, i, j, 0.0);
        }
    }*/
    
    /* Current nonzero random matrix, generates one value in each row.
     * This has not been correlated.
     * Either this code block or the one below should be commented out.
     */
    j = 0;
    for(i=0; i<nonzero_N; ++i) {
        if(j == k) j = 0;
        gsl_matrix_set(randn, i, j, gsl_ran_gaussian(rng, gsl_vector_get(stddev, j)));
        ++j;
    }
    
    /* Current nonzero random matrix, generates at least one value in each row more dependent on
     * a random distribution and density. This data has not been correlated.
     * Either this code block or the one above should be commented out.
     */
    /*rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
    randn = gsl_matrix_calloc(nonzero_N, k);
    for(i=0; i<nonzero_N; ++i) {
        p = 0.0;
        while(p == 0.0)
            for(j=0; j<k; ++j) {
                if(gsl_rng_uniform(rng) <= density){
                    gsl_matrix_set(randn, i, j, gsl_ran_gaussian(rng, gsl_vector_get(stddev, j)));
                    ++p;
                }
            }
        
    }*/
    
    /* Print out the pre-reward matrix, should be removed */
    /*printf("Non-zero random matrix without reshaping. (Should be removed)\n");
    for(i=0;i<nonzero_N;++i){
        for(j=0;j<k;++j){
            printf("%8.4f", gsl_matrix_get(randn, i, j));
        }
        printf("\n");
    }*/
    
    /* Sparsisty test code, needs cleanup, also not in use */
    /*double div;
    int t;
    double temp;
    gsl_matrix *maskc = gsl_matrix_calloc(N,k);
    
    for (i = 0; i < N; ++i) {
        for (j = 0; j < k; ++j) {
            if (gsl_rng_uniform(rng) <= density) gsl_matrix_set(unmapped_samples, i, j, 1.0);
        }
    }
    
    for (i = 0; i < N; ++i) {
        
        for (j = 0; j < k; ++j) {
            div = 1.0;
            p = density;
            for (t = 0; t < k; ++t) {
                if ( gsl_matrix_get(unmapped_samples, i, t) == 1.0) {
                    div += 1.0;
                    temp = gsl_matrix_get(r, j, t);
                    p = temp < 0.0 ? p - temp : p + temp;
                }
            }
            if (gsl_rng_uniform(rng) <= p/div) gsl_matrix_set(unmapped_samples, i, j, 1.0);
            gsl_matrix_set(maskc,i,j,p/div);
        }
    }
    
    gsl_matrix *c = pearson_correlation(unmapped_samples);
    
    printf("Printout of calculation matrix for sparsity:\n");
    for(i=0;i<N;++i){
        for(j=0;j<k;++j){
            printf("%8.4f:%2.1f", gsl_matrix_get(maskc, i, j), gsl_matrix_get(unmapped_samples,i,j));
        }
        printf("\n");
    }*/
    /* End of test code for sparsisty */
    
    /* multiply it by the cholesky decomposition of R */
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, randn, T, 0.0, unmapped_samples);

    /* add in the means */
    gsl_matrix_add(unmapped_samples, v);
    
    /* Check for breach of lower bound */
    if (lower != NULL) {
        for(j=0;j<k;++j){
            p = gsl_vector_get(lower,j);
            for(i=0;i<nonzero_N;++i){
                if (p > gsl_matrix_get(unmapped_samples,i,j))
                    gsl_matrix_set(unmapped_samples,i,j,p);
            }
        }
    }
    
    /* Check for breach of upper bound */
    if (upper != NULL) {
        for(j=0;j<k;++j){
            p = gsl_vector_get(upper,j);
            for(i=0;i<nonzero_N;++i){
                if (p < gsl_matrix_get(unmapped_samples,i,j))
                    gsl_matrix_set(unmapped_samples,i,j,p);
            }
        }
    }
    
    /* Generate zeros matrix with N elements.
     * If
     */
    for(i = 0; i < N; ++i){
        zeros[i] = 0;
    }
    int g = (int)floor(gsl_rng_uniform(rng)*N);
    
    /* Generate a mapping order to map unmapped_samples to the return matrix *samples
     * and simutaneously perform said mapping.
     */
    for(i = 0; i < nonzero_N; ++i){
        /*mapping_array[i] = (int)floor(gsl_rng_uniform(rng)*N);
        
        // This while loop maintains the necessary one-to-one property of the mapping function
        j = 0;
        while(j < i){
            if(mapping_array[i] == mapping_array[j]){
                mapping_array[i] = (int)floor(gsl_rng_uniform(rng)*N);
                j = 0;
            } else ++j;
        }*/
        
        while(zeros[g]) g = (int)floor(gsl_rng_uniform(rng)*N);
        
        zeros[g] = 1;
        mapping_array[i] = g;
        
        for(j = 0; j < k; ++j)
            gsl_matrix_set(*samples,mapping_array[i],j,gsl_matrix_get(unmapped_samples,i,j));
            
    }
    
    /*test-code*/
    
    /* Calculate and print the pearson correlation of the generated rewards, should be removed */
    gsl_matrix *c = pearson_correlation(unmapped_samples);
    print_gsl_matrix(stdout, c, ";;  ", "Calculated correlation matrix");
    
    /* Calculate and print the sum absolute error per permutation, should be removed */
    int u, w;
    double ans = 0;
    double t,x,y;
    for (u = 0; u < k; ++u) {
       for (w = u+1; w < k; ++w){
           x = gsl_matrix_get(r,u,w);
           y = gsl_matrix_get(c,u,w);
           t = x - y;
           ans = t<0.0 ? ans-t : ans+t;
       }
    }
    u = 1;//(k*k-k)/2;
    printf("\nAbsolute error-sum: %8.4f\n",ans/u);
    
    /*test-code end*/
    
    /* Generate a mapping order to map unmapped_samples to the return matrix *samples
     * and simutaneously perform said mapping.
     * Repeat for test-code
     */
    /*for(i = 0; i < nonzero_N; ++i){
        mapping_array[i] = (int)floor(gsl_rng_uniform(rng)*N);
        
        // This while loop maintains the necessary one-to-one property of the mapping function
        j = 0;
        while(j < i){
            if(mapping_array[i] == mapping_array[j]){
                mapping_array[i] = (int)floor(gsl_rng_uniform(rng)*N);
                j = 0;
            } else ++j;
        }
        
        for(j = 0; j < k; ++j)
            gsl_matrix_set(*samples,mapping_array[i],j,gsl_matrix_get(unmapped_samples,i,j));
            
    }*/
    
    /*printf("\nOrder: ");
    for(i=0;i<nonzero_N;++i) printf("%d, ",mapping_array[i]);
    printf("\n");*/
    
    
    /* Print out the pre-reward matrix */
    /*printf("Non-zero reward matrices without reshaping. (Should be removed)\n");
    for(i=0;i<nonzero_N;++i){
        for(j=0;j<k;++j){
            printf("%8.4f", gsl_matrix_get(unmapped_samples, i, j));
        }
        printf("\n");
    }*/
    
    
    /* Print out the reward matrix */
    /*printf("Reward matrices without reshaping. (Should be removed)\n");
    for(i=0;i<N;++i){
        for(j=0;j<k;++j){
            printf("%8.4f", gsl_matrix_get(*samples, i, j));
        }
        printf("\n");
    }*/
    
    /* Test code that checks end result sparsity, needs clean up */
    /*int counter;
    
    printf("\n\n");
    for(i=0;i<k;++i){
        counter = 0;
        for(j=0;j<N;++j){
            if(gsl_matrix_get(*samples, j, i)==0.0) ++counter;
        }
        printf("%d, %f\n",counter,(double)counter/N);
    }*/
    /* End of test code */
    
    /*printf("\nSilly code\n\n");
    
    for(i=0;i<N;++i){
        for(j=0;j<k;++j){
            gsl_matrix_set(*samples, i, j, i);
            printf("%8.4f, ", gsl_matrix_get(*samples, i, j));
        }
        printf("\n");
    }
    
    printf("\n");*/
    
    gsl_matrix *fuu = gsl_matrix_alloc(N,k);
    int max_iter = 12;
    int curr_iter = 0;
    int index, kt;
    double sum, div, temp;
    p = 1.0;
    
    while(p > 0.01 && curr_iter < max_iter){
        gsl_matrix_memcpy(fuu, *samples);
        p = 0.0;
        
        for(i = 0; i < n; ++i){
            for(j = 0; j < m; ++j){
                index = i*m + j;
                for(kt = 0; kt < k; ++kt){
                    div = 1.0;
                    sum = gsl_matrix_get(fuu, index, kt);
                    
                    if(i != 0 && j != 0){
                        sum += gsl_matrix_get(fuu, (index - m - 1), kt);
                        div += 1.0;
                    }
                    
                    if(i != 0){
                        sum += gsl_matrix_get(fuu, (index - m), kt);
                        div += 1.0;
                    }
                    
                    if(i != 0 && j != m-1){
                        sum += gsl_matrix_get(fuu, (index - m + 1), kt);
                        div += 1.0;
                    }
                    
                    if(j != 0){
                        sum += gsl_matrix_get(fuu, (index - 1), kt);
                        div += 1.0;
                    }
                    
                    if(j != m-1){
                        sum += gsl_matrix_get(fuu, (index + 1), kt);
                        div += 1.0;
                    }
                    
                    if(i != n-1 && j != 0){
                        sum += gsl_matrix_get(fuu, (index + m - 1), kt);
                        div += 1.0;
                    }
                    
                    if(i != n-1){
                        sum += gsl_matrix_get(fuu, (index + m), kt);
                        div += 1.0;
                    }
                    
                    if(i != n-1 && j != m-1){
                        sum += gsl_matrix_get(fuu, (index + m + 1), kt);
                        div += 1.0;
                    }
                    
                    gsl_matrix_set(*samples, index, kt, sum/div);
                    temp = fabs(gsl_matrix_get(fuu, index, kt) - gsl_matrix_get(*samples, index, kt));
                    if(temp > p) p = temp;
                }
            }
        }
        ++curr_iter;
    }
    
    for(i = 0; i < N; ++i){
        if(zeros[i] == 0)
            for(j = 0; j < k; ++j){
                if(fabs(gsl_matrix_get(*samples, i, j)) < 0.01)
                   gsl_matrix_set(*samples, i, j, 0.0);
            }
    }
    
    printf("\n%d iterations",curr_iter);
    
    gsl_rng_free(rng);
    gsl_matrix_free(randn);
    gsl_matrix_free(v);
    gsl_matrix_free(T);
    gsl_matrix_free(unmapped_samples);
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

/* Find matrix subject's pearson correlation matrix, 
 * uses subject's columns as seperate entities to correlate
 * 
 * Usage:  gsl_matrix *pearson = pearson_correlation(subject);
 * Input:  gsl matrix pointer subject with dimensions Nxk
 * Output: pointer to kxk symmetric gsl matrix pearson where element
 *         i, j is the pearson correlation of column i and j of subject
 *         if subject is NULL then so is pearson
 */
gsl_matrix* pearson_correlation(gsl_matrix* subject)
{
    if ( subject == NULL) return subject;
    
    int i;
    int j;
    double p;
    gsl_vector a;
    gsl_vector b;
    int k = subject->size2;
    int N = subject->size1;
    
    gsl_matrix* pearson = gsl_matrix_alloc(k, k);
    gsl_matrix_set_identity(pearson);
    
    for (i = 0; i < k-1; ++i) {
        a = gsl_matrix_column(subject, i).vector;
        
        for (j = i+1; j < k; ++j) {
            b = gsl_matrix_column(subject,j).vector;
            p = gsl_stats_correlation(a.data, a.stride, b.data, b.stride, N);
            gsl_matrix_set(pearson,i,j,p);
            gsl_matrix_set(pearson,j,i,p);
        }
    }
    
    return pearson;
}

/* Generates a MATLAB script file name.m containing k and all reward matrices.
 * Running the script will create variable $name$_count = k,
 * and k matrices named name_ti = reward matrix for task i
 * where i in [0..k-1].
 * Currently also generates MATLAB code at the end of the script that displays
 * the rewards as heat maps.
 *
 * Usage:  print_to_matlab(n,m,k,rewards,name);
 * Input:  n,m,k are integers
 *         rewards is a nmXk gsl_matrix pointer
 *         name is a null terminated string, if NULL then defaults to "A"
 * Output: Only sideeffects, the creation/changing of file name.m
 */
void print_to_matlab(int n, int m, int k, gsl_matrix *rewards, char* name)
{
    FILE *fs = NULL;
    int num_cols = rewards->size2;
    gsl_vector_view col;
    int i, state, action, index;
    
    if(name == NULL){
        fprintf(stderr, "Warning: Output file and matrix name unspecified, using A as default\n");
        name = "A";
    }
    
    char *filename = (char*)malloc(strlen(name) + 3);
    strcpy(filename, name);
    strcat(filename, ".m");
    
    if((fs = fopen(filename, "w"))){
        printf("Printing to file %s\n", filename);
        fprintf(fs, "%s_count = %d;\n\n", name, k);
        
        for(i=0; i<num_cols; ++i) {
            fprintf(fs, "%s_t%d = [\n\t", name, i + 1);
            col = gsl_matrix_column(rewards, i);
            index = 0;
            for(state=0; state<n; ++state) {
                for(action=0; action<m; ++action) {
                    fprintf(fs, "%8.12f,", gsl_vector_get(&col.vector, index++));
                }
                fprintf(fs,";\n\t");
            }
            fprintf(fs,"];\n\n");
        }
        /* Test code, should probably be dropped */
        
        int sq = (int)(ceil(sqrt((double)k)));
        int j;
        char *fn = "imagesc";
        
        fprintf(fs, "figure(1); cla;\n\n");
        
        index = 0;
        for(i = 1; i < sq + 1; ++i){
            for(j = 1; j < sq + 1; ++j){
                if(index < k){
                    fprintf(fs, "subplot(%d,%d,%d);\n",sq,sq,++index);
                    fprintf(fs, "CLIM = max(max(abs(%s_t%d)));\n", name, index);
                    fprintf(fs, "%s(%s_t%d, [-CLIM, CLIM])\n", fn, name, index);
                    //printf(fs, "%s(abs(%s_t%d))\n", fn, name, index);
                    fprintf(fs, "title('%s\\_t%d');\naxis image;\n\n", name, index);
                }
            }
        }
        
        fprintf(fs, "colormap(jet)\n\n");
        
        /* Test code ends */
        
        fclose(fs);
    } else
        fprintf(stderr, "Error: Could not open %s\n", filename);
        
    free(filename);
}
