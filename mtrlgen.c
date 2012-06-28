/*
 * mtrlgen.c
 *
 * Problem generator for random multi-task reinforcement learning problems.
 *
 * Generates instances with $k$ tasks, each correlated as specified by a given
 * covariance matrix. Note that the covariance matrix must be positive-definite.
 */

/* 
 * Copyright 2012 Deon Garrett <deon@iiim.is>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


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

void populate_subopts(char** tokens, int k);
void free_subopt_tokens(char*** tokens, int sz);
void print_usage();
void print_gsl_matrix(gsl_matrix* m);
int  generate_instance(gsl_matrix** samples, int n, int m, int k, gsl_matrix* r, 
		       gsl_vector *mean, gsl_vector *stddev, double density, unsigned long seed, gsl_vector *lower, gsl_vector *upper);
unsigned long get_seed();
gsl_matrix* pearson_correlation(gsl_matrix* r);
void print_to_matlab(int n, int m, int k, gsl_matrix *rewards, char* name);

/*
 * create an array of suboption names for the correlation pairs.
 *
 * Option names are created according to the following scheme: given
 * k tasks, (numbered 1 through k), a suboption name is created for
 * each pair (i, j) such that 1 <= i <= k and i+1 <= j <= k. For
 * example, with four tasks, the tokens array will contain the strings
 * ["12", "13", "14", "23", "24", "34"].
 *
 * Note that the tokens array must be allocated in the caller with
 * enough space to contain k*(k-1)/2 options.
 */
void populate_subopts(char** tokens, int k) 
{
    int i, j;
    int index=0;

    /* fill out the tokens */
    for(i=1; i<=k; ++i) {
        for(j=i+1; j<=k; ++j) {
            sprintf(tokens[index++], "%1d%1d", i, j);
        }
    }
    tokens[index] = NULL;
}

/* free the memory used by the suboption array */
void free_subopt_tokens(char*** tokens, int sz)
{
    int i;
    for(i=0; i<sz; ++i) {
        free((*tokens)[i]);
    }
    free(*tokens);
}

/* debugging function to print a matrix */
void print_gsl_matrix(gsl_matrix* m)
{
    int i, j;
    for(i=0; i<m->size1; ++i) {
        printf(";; ");
        for(j=0; j<m->size2; ++j) {
            printf("%10.6f ", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/* print a brief help message */
void print_usage()
{
    printf("\tusage: generator [-h] -n STATES -m ACTIONS -k TASKS [-d density] [-s seed] [-rXY rho] ...\n\n");
    printf("The parameters -rXY (where X and Y are single-digit integers [1..9]) specify\n"
           "the correlation coefficient between tasks X and Y. By definition, this coefficient\n"
           "is 1.0 where X=Y. For any other (X,Y) pair, if the given parameter is not\n"
           "specified, the coefficient is assumed to be 0.0 (i.e., the tasks are uncorrelated.\n\n"
           "Example: \n"
           "\tmtrlgen -n100 -m5 -k3 -r12=0.25,13=-0.4,23=0.1\n\n"
           "produces a problem with 100 states, 5 actions, and 3 tasks, where the tasks are\n"
           "related by the correlation matrix\n\n"
           "\t[ 1.0    0.25  -0.4\n"
           "\t  0.25   1.0    0.1\n"
           "\t -0.4    0.1    1.0 ]\n\n");
}

int generate_instance(gsl_matrix** samples, int n, int m, int k, gsl_matrix* r, 
		      gsl_vector *mean, gsl_vector *stddev, double density, unsigned long seed, gsl_vector *lower, gsl_vector *upper) 
{
    /* the total number of samples for each task is nm (states * actions).
     * we'll reshape this into a matrix form when we're done. */
    const int N = m*n;
    gsl_matrix* v;
    gsl_matrix* T;
    gsl_matrix* randn;
    gsl_matrix *unmapped_samples;
    gsl_rng* rng;
    int i;
    int j;
    double p;
    int nonzero_N = (int)ceil(N*density);
    
    /* Memory allocation */
    unsigned int mapping_array[nonzero_N];
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
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
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
    
    /* Generate an mapping order to map unmapped_samples to the return matrix *samples
     * and simutaneously perform said mapping.
     */
    for(i = 0; i < nonzero_N; ++i){
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
            
    }
    
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

/* print the instance in the defined output format */
void print_instance(int n, int m, int k, gsl_matrix* rewards)
{
    /*
     * rewards is at this point an nmXk matrix where each of the k columns
     * are interrelated according to the given correlation matrix.
     *
     * To convert this to a normalized problem form, we need to take the
     * values for each task, currently stored in a single column, and reshape
     * them back into an nXm matrix.
     */
    int num_cols = rewards->size2;
    gsl_vector_view col;
    /*gsl_matrix_view task;*/
    int i, state, action, index;
    
    printf("%d\n%d\n%d\n\n", n, m, k);
    for(i=0; i<num_cols; ++i) {
        col = gsl_matrix_column(rewards, i);
        /*task = gsl_matrix_view_vector(&col.vector, n, m);*/
        index = 0;
        for(state=0; state<n; ++state) {
            for(action=0; action<m; ++action) {
                printf("%8.4f", gsl_vector_get(&col.vector, index++));
            }
            printf("\n");
        }
        printf("\n");
    }
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
                    fprintf(fs, "%8.4f,", gsl_vector_get(&col.vector, index++));
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
                    fprintf(fs, "%s(%s_t%d)\n", fn, name, index);
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

/* main
 *
 * parses the command line options and calls the generator
 */
int main(int argc, char** argv)
{
    unsigned int n = 0;
    unsigned int m = 0;
    unsigned int k = 0;
    gsl_matrix*  r = NULL;
    gsl_matrix*  rewards = NULL;
    int opt;
    char** tokens = NULL;
    char* matlab_name = NULL;
    unsigned long seed = 0;
    int seed_given = 0;
    int num_toks = 0;
    gsl_vector *mean = NULL;
    gsl_vector *stddev = NULL;
    gsl_vector *lower = NULL;
    gsl_vector *upper = NULL;
    double density = 0.1;
    
    /* parse command line options */
    while((opt = getopt(argc, argv, "n:m:k:r:o:v:s:d:h:l:u:q:")) != -1) {
        switch(opt) {
        case 'n':
            n = atol(optarg);
            break;
            case 'm':
            m = atol(optarg);
            break;
        case 'k':
        {
            const int maxstrlen = 2;
            int i;

            /* get number of tasks */
            k = atoi(optarg);

            /* initialize the covariance matrix */
            r = gsl_matrix_calloc(k, k);
            gsl_matrix_set_identity(r);
            
            /* initialize the means and standard deviation vectors */
            mean = gsl_vector_calloc(k);
            stddev = gsl_vector_alloc(k);
            gsl_vector_set_all(stddev, 1.0);
            
            /* also set up the tokens array for suboption parsing */
            num_toks = (int)(k*(k-1)/2);
            tokens = (char**)malloc((num_toks+1)*sizeof(char*));
            for(i=0; i<num_toks; ++i) {
                tokens[i] = (char*)malloc((maxstrlen+1)*sizeof(char));
            }
            populate_subopts(tokens, k);
            break;
        }
        case 'r': 
        {
            char* subopts = optarg;
            char* value;
            while(*subopts != '\0') {
                int suboptpos = getsubopt(&subopts, tokens, &value);
                if(suboptpos < 0) {
                    fprintf(stderr, "Error: invalid r-specification. All -rXY pairs must have 0<X<Y.\n");
                    goto cleanup;
                }
                int xy = atoi(tokens[suboptpos]);
                int x = xy/10;
                int y = xy%10;
                assert(x<y);
                gsl_matrix_set(r, x-1, y-1, atof(value));
                gsl_matrix_set(r, y-1, x-1, atof(value));
            }
            break;
        }
        case 's':
            seed_given = 1;
            seed = (unsigned long)atol(optarg);
            break;
        case 'o':
        {
            char *start = optarg;
            char *end;
            double value;
            int i = 0;
            while (i<k) {
                value = strtod(start, &end);
                start = &(end[1]);
                gsl_vector_set(mean,i++,value);
            }
            break;
        }
        case 'v':
        {
            char *start = optarg;
            char *end;
            double value;
            int i = 0;
            while (i<k) {
                value = strtod(start, &end);
                start = &(end[1]);
                gsl_vector_set(stddev,i++,value);
            }
            break;
        }
        case 'l':
        {
            char *start = optarg;
            char *end;
            double value;
            int i = 0;
            lower = gsl_vector_alloc(k);
            while (i<k) {
                value = strtod(start, &end);
                start = &(end[1]);
                gsl_vector_set(lower,i++,value);
            }
            if (upper != NULL){
                for (i=0;i<k;++i){
                    if (gsl_vector_get(lower,i)>gsl_vector_get(upper,i)) {
                        fprintf(stderr, "Error: lower bound for task %d is greater than it's upper bound\n",i+1);
                        goto cleanup;
                    }
                }
            }
            break;
        }
        case 'u':
        {
            char *start = optarg;
            char *end;
            double value;
            int i = 0;
            upper = gsl_vector_alloc(k);
            while (i<k) {
                value = strtod(start, &end);
                start = &(end[1]);
                gsl_vector_set(upper,i++,value);
            }
            if (lower != NULL){
                for (i=0;i<k;++i){
                    if (gsl_vector_get(lower,i)>gsl_vector_get(upper,i)) {
                        fprintf(stderr, "Error: lower bound for task %d is greater than it's upper bound\n",i+1);
                        goto cleanup;
                    }
                }
            }
            break;
        }
        case 'd':
            density = atof(optarg);
            /* TODO: make it so that at least one value is non-zero */
            assert(density>0.01);
            break;
        case 'q':
            matlab_name = optarg;
            break;
        case 'h':
        default:
            print_usage();
            return EXIT_FAILURE;
        }
    }
    
    /* sanity check */
    if(!(n && m && k)) {
        fprintf(stderr, "Error: required parameters n, m, and k not set. Run mtrlgen -h for usage information.\n");
        goto cleanup;
    }

    /* if no random number generator seed was given, pick one */
    if(!seed_given) {
        seed = get_seed();
    }
    
    /* generate instance */
    if(generate_instance(&rewards, n, m, k, r, mean, stddev, density, seed, lower, upper) != 0) {
        fprintf(stderr, "Error generating instance. Verify that covariance matrix is positive-definite.\n");
        goto cleanup;
    }
    
    /* print instance */
    print_instance(n, m, k, rewards);
    
    /* print a footer identifying generator parameters */
    printf("\n\n");
    printf(";; states=%d, actions=%d, tasks=%d\n", n, m, k);
    printf(";; User correlation matrix=\n");
    print_gsl_matrix(r);
    
    /* Calculate and print the pearson correlation of the generated rewards, should be removed */
    gsl_matrix *c = pearson_correlation(rewards);
    printf("\n;; Calculated correlation matrix=\n");
    print_gsl_matrix(c);
    
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
    
    /* Print to matlab file, */
    if(matlab_name) print_to_matlab(n,m,k,rewards,matlab_name);
    
cleanup:
    if(tokens) {
        free_subopt_tokens(&tokens, num_toks);
    }
    if(r) {
        gsl_matrix_free(r);
    }
    if(rewards) {
        gsl_matrix_free(rewards);
    }
    if(mean) {
        gsl_vector_free(mean);
    }
    if(stddev) {
        gsl_vector_free(stddev);
    }
    if(lower) {
        gsl_vector_free(lower);
    }
    if(upper) {
        gsl_vector_free(upper);
    }
    if(c) {
        gsl_matrix_free(c);
    }
    
    return 0;
}
