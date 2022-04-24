#include <stdlib.h>
#include "gn.h"

#define TRUE 1
#define FALSE 0

void run_monte_carlo(const int *i_exp,
                     const int *i_node,
                     const int *outcome,
                     const double *value,
                     const int *is_perturbation,
                     int len_i_exp)
{
    // prepare data
    struct experiment_set experiment_set;
    struct network net;

    // Set parameters
    int n = len_i_exp;
    // int n = 9;
    double T_lo = 0.001;
    double T_hi = 2.0;
    double target_score = 0;
    int n_thread = 1;
    int max_states = 10;
    int exchange_interval = 1000;
    int adjust_move_size_interval = 7001;
    int n_cycles = 1000000;
    int max_parents = 10;
    int n_write = 10;
    FILE *output_file = fopen("../runs/debug.txt", "w");

    // initialize data structures
    experiment_set_init(&experiment_set, n, i_exp, i_node, outcome, value, is_perturbation);
    printf("Intialized experiment\n");
    network_init(&net, experiment_set.n_node, max_parents);
    printf("Intialized network\n");
    network_randomize_parents(&net);
    printf("Randomized parents\n");
    network_set_outcomes_to_null(&net);
    printf("Set outcomes to null\n");

    // network monte carlo
    printf("Running score with parameters n_cycles=%d n_write=%d T_lo=%lf T_high=%lf n_thread=%d target_score=%lf exchange_interval=%d adjust_move_size_interval=%d max_states=%d\n",
        n_cycles, n_write, T_lo, T_hi, n_thread, target_score, exchange_interval, adjust_move_size_interval, max_states);
    double score = network_monte_carlo(
        &net,
        &experiment_set,
        n_cycles,
        n_write,
        T_lo,
        T_hi,
        output_file,
        n_thread,
        target_score,
        exchange_interval,
        adjust_move_size_interval,
        max_states);
    printf("Finished running score, obtained value %lf\n", score);
}

void test()
{
    const int i_exp[] = {0, 0, 0, 0, 0, 0, 0, 0, 0,
                         1, 1, 1, 1, 1, 1, 1, 1, 1,
                         2, 2, 2, 2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 3, 3, 3, 3, 3,
                         4, 4, 4, 4, 4, 4, 4, 4, 4,
                         5, 5, 5, 5, 5, 5, 5, 5, 5,
                         6, 6, 6, 6, 6, 6, 6, 6, 6,
                         7, 7, 7, 7, 7, 7, 7, 7, 7,
                         8, 8, 8, 8, 8, 8, 8, 8, 8,
                         9, 9, 9, 9, 9, 9, 9, 9, 9,
                         10, 10, 10, 10, 10, 10, 10, 10, 10,
                         11, 11, 11, 11, 11, 11, 11, 11, 11,
                         12, 12, 12, 12, 12, 12, 12, 12, 12,
                         13, 13, 13, 13, 13, 13, 13, 13, 13,
                         14, 14, 14, 14, 14, 14, 14, 14, 14,
                         15, 15, 15, 15, 15, 15, 15, 15, 15,
                         16, 16, 16, 16, 16, 16, 16, 16, 16,
                         17, 17, 17, 17, 17, 17, 17, 17, 17};

    const int i_node[] = {0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2,
                          0, 0, 0, 1, 1, 1, 2, 2, 2};

    const int outcome[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1,
                           -1, 0, 1, -1, 0, 1, -1, 0, 1};

    const double value[] = {2, 1, 0, 1, 0, 1, 2, 1, 0,
                            1, 0, 1, 2, 1, 0, 0, 1, 2,
                            1, 0, 1, 1, 0, 1, 2, 1, 0,
                            0, 1, 2, 1, 0, 1, 0, 1, 2,
                            1, 0, 1, 0, 1, 2, 2, 1, 0,
                            1, 0, 1, 1, 0, 1, 0, 1, 2,
                            2, 1, 0, 1, 0, 1, 2, 1, 0,
                            0, 1, 2, 1, 0, 1, 0, 1, 2,
                            1, 0, 1, 2, 1, 0, 0, 1, 2,
                            1, 0, 1, 0, 1, 2, 2, 1, 0,
                            2, 1, 0, 2, 1, 0, 1, 0, 1,
                            0, 1, 2, 0, 1, 2, 1, 0, 1,
                            2, 1, 0, 0, 1, 2, 2, 1, 0,
                            0, 1, 2, 2, 1, 0, 0, 1, 2,
                            2, 1, 0, 1, 0, 1, 0, 1, 2,
                            0, 1, 2, 1, 0, 1, 2, 1, 0,
                            1, 0, 1, 2, 1, 0, 2, 1, 0,
                            1, 0, 1, 0, 1, 2, 0, 1, 2};

    const int is_perturbation[] = {TRUE, TRUE, TRUE,
                                   FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
                                   FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE,
                                   FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE,
                                   TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
                                   FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE,
                                   FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE,
                                   TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE,
                                   TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE,
                                   FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                                   FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                                   TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE,
                                   TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE,
                                   TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE,
                                   TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE,
                                   TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE,
                                   TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, TRUE, TRUE,
                                   FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE,
                                   FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE};

    int len_i_exp = sizeof(i_exp) / sizeof(i_exp[0]); 
    run_monte_carlo(&i_exp[0], &i_node[0], &outcome[0], &value[0], &is_perturbation[0], len_i_exp);
}

int main()
{
    test();
    return 0;
}