#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <Rmath.h>

#include "array.h"

#define ADJUST_MOVE_SIZE_INTERVAL 7001
#define EXCHANGE_INTERVAL 1000

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "util.h"
#include "gn.h"

#define UNDEFINED 9
#define LARGE_SCORE 1e10

static int state_from_sym(char c)
{
  switch (c) {
  case '-': 
    return -1;
  case '.':
    return 0;
  case '+':
    return 1;
  case '?':
    return UNDEFINED;
  default:
    die("state_from_sym: unknown character: '%c'.  must be one of - . + ?", c);
  }
  return 0;
}

static char sym_from_state(int s)
{
  switch (s) {
  case -1:
    return '-';
  case 0:
    return '.';
  case 1:
    return '+';
  case UNDEFINED:
    return '?';
  default:
    die("sym_from_state: unknown state: %d", s);
  }
  return 0;
}

static void read_line(FILE *f, char *buf, int n)
{
  if (!fgets(buf, n, f))
    die("read_line: unexpected end of file");
  if (strlen(buf) >= n)
    die("read_line: line too long");
}

static double uniform_random_from_0_to_1_exclusive()
{
  /* return (double) random() / ((double) RAND_MAX + 1.0); */
  return unif_rand();
}

static int random_int_inclusive(int a, int b)
{
  return (int) floor((b-a+1)*uniform_random_from_0_to_1_exclusive()) + a;
}

#define MAX_LINE (1024 + MAX_NODES)

unsigned three_to_the(unsigned n)
{
  unsigned a = 1;
  while (n-- > 0)
    a *= 3;
  return a;
}

void network_init(network_t n, int n_node, int max_parents)
{
  n->n_node = n_node;
  if (max_parents > n_node - 1)
    max_parents = n_node - 1;
  n->n_parent = max_parents;
  n->n_outcome = three_to_the(max_parents);
  n->parent = int_array2D_new(n->n_node, n->n_parent);
  n->outcome = int_array2D_new(n->n_node, n->n_outcome);
  int i;
  for (i = 0; i < n_node; i++) {
    int j;
    for (j = 0; j < n->n_parent; j++)
      n->parent[i][j] = (i+j+1)%n_node;
    for (j = 0; j < n->n_outcome; j++)
      n->outcome[i][j] = state_from_sym('.');
  }
}  

void network_delete(network_t n)
{
  int_array2D_delete(n->parent);
  int_array2D_delete(n->outcome);
}

static int intcmp(const void *a, const void *b)
{
  if (*(const int *) a < *(const int *) b)
    return -1;
  if (*(const int *) a > *(const int *) b)
    return 1;
  return 0;
}

void network_write(FILE *f, const network_t n)
{
  int i;
  for (i = 0; i < n->n_node; i++) {
    int ip;
    for (ip = 0; ip < n->n_parent; ip++)
      fprintf(f, "%d ", n->parent[i][ip]);
    if (n->n_parent > 0) {
      for (ip = 0; ip < n->n_outcome; ip++)
	fprintf(f, "%c", sym_from_state(n->outcome[i][ip]));
    }
    fprintf(f, "\n");
  }
}

static int repetition_found(const trajectory_t t)
{
  return t->repetition_end > 0;
}

static void advance(const network_t n, trajectory_t traj, int i_state)
{
  const int n_node = n->n_node;
  /* find new state */
  int *si = &traj->state[i_state][0];
  const int *si1 = &traj->state[i_state - 1][0];
  int i_node;
  for (i_node = 0; i_node < n_node; i_node++) {
    if (traj->is_persistent[i_node] || n->n_parent == 0) {
      si[i_node] = si1[i_node];
    } else {
      int ip, a = 0;
      for (ip = n->n_parent - 1; ip >= 0; ip--) {
	a *= 3;
	a += si1[n->parent[i_node][ip]] + 1;
      }
      si[i_node] = n->outcome[i_node][a];
    }
  }
}

static void check_for_repetition(trajectory_t traj, int i_state)
{
  const int n_node = traj->n_node;
  int i_node, j_state;
  const int *si = &traj->state[i_state][0];
  for (j_state = 0; j_state < i_state; j_state++) {
    const int *sj = &traj->state[j_state][0];
    for (i_node = 0; i_node < n_node; i_node++)
      if (si[i_node] != sj[i_node])
	break;
    if (i_node < n_node)
      continue;
    /* repetition found - create summary */
    traj->repetition_start = j_state;
    traj->repetition_end = i_state;
    for (i_node = 0; i_node < traj->n_node; i_node++) {
      int k, visited_plus = 0, visited_minus = 0;
      for (k = traj->repetition_start; k <= traj->repetition_end; k++)
	if (traj->state[k][i_node] == 1)
	  visited_plus = 1;
	else if (traj->state[k][i_node] == -1)
	  visited_minus = 1;
      if (visited_plus && visited_minus)
	traj->steady_state[i_node] = UNDEFINED;
      else if (visited_plus)
	traj->steady_state[i_node] = 1;
      else if (visited_minus)
	traj->steady_state[i_node] = -1;
      else
	traj->steady_state[i_node] = 0;
    }
    return;
  }
  /* no repetition found */
  traj->repetition_start = 0;
  traj->repetition_end = 0;
  for (i_node = 0; i_node < n_node; i_node++)
    traj->steady_state[i_node] = UNDEFINED;
}

static double score_for_state(const experiment_t e, int i, int s)
{
  return e->score[i][s+1];
}

static void set_score_for_state(const experiment_t e, int i, int s, double val)
{
  e->score[i][s+1] = val;
}

static int most_probable_state(const experiment_t e, int i)
{
  int min_s = -1;
  double min = score_for_state(e,i,-1);
  int s;
  for (s = 0; s <= 1; s++)
    if (score_for_state(e,i,s) < min) {
      min_s = s;
      min = score_for_state(e,i,s);
    }
  return min_s;
}

static double score_for_most_probable_state(const experiment_t e, int j)
{
  return score_for_state(e,j,most_probable_state(e,j));
}

static void init_trajectory(trajectory_t t, const experiment_t e, int n_node)
{
  t->n_node = n_node;
  int i;
  for (i = 0; i < t->n_node; i++) {
    t->is_persistent[i] = 0;
    t->state[0][i] = 0;
  }
  t->repetition_start = t->repetition_end = 0;
  for (i = 0; i < e->n_perturbed; i++) {
    const int j = e->perturbed[i];
    t->is_persistent[j] = 1;
    t->state[0][j] = most_probable_state(e,j);
  }
}

void experiment_set_read_as_csv(FILE *f, experiment_set_t e)
{
  int i_exp;
  e->n_experiment = 0;
  e->n_node = 0;
  for (i_exp = 0; i_exp < MAX_EXPERIMENTS; i_exp++)
    e->experiment[i_exp].n_perturbed = 0;
  char buf[MAX_LINE];
  read_line(f, buf, MAX_LINE); /* skip field description line */
  while (!end_of_file(f)) {
    read_line(f, buf, MAX_LINE);
    int i_node, is_perturbed, outcome;
    double val;
    if (sscanf(buf, "%d, %d, %d, %lf, %d", &i_exp, &i_node, &outcome, &val, &is_perturbed) != 5)
      die("experiment_read: expecting i_exp i_node outcome value is_perturbed, found %s", buf);
    if (outcome < -1 || outcome > 1)
      die("experiment_read: outcome %d is out of range", outcome);
    if (i_exp < 0 || i_exp >= MAX_EXPERIMENTS)
      die("experiment_read: i_exp=%d is out of range, MAX_EXPERIMENTS=%d", i_exp, MAX_EXPERIMENTS);
    if (i_node < 0 || i_node >= MAX_NODES)
      die("experiment_read: i_node=%d is out of range, MAX_NODES=%d", i_exp, MAX_NODES);
    experiment_t en = &e->experiment[i_exp];
    set_score_for_state(en, i_node, outcome, val);
    if (is_perturbed)
      en->perturbed[en->n_perturbed++] = i_node;
    if (i_exp >= e->n_experiment)
      e->n_experiment = i_exp + 1;
    if (i_node >= e->n_node)
      e->n_node = i_node + 1;
  }
}

void experiment_set_init(experiment_set_t e, 
			 int n,
			 const int *i_exp, 
			 const int *i_node, 
			 const int *outcome,
			 const double *val,
			 const int *is_perturbation)
{
  e->n_experiment = 0;
  e->n_node = 0;
  int j_exp;
  for (j_exp = 0; j_exp < MAX_EXPERIMENTS; j_exp++)
    e->experiment[j_exp].n_perturbed = 0;
  int i;
  for (i = 0; i < n; i++) {
    experiment_t en = &e->experiment[i_exp[i]];
    set_score_for_state(en, i_node[i], outcome[i], val[i]);
    if (is_perturbation[i])
      en->perturbed[en->n_perturbed++] = i_node[i];
    if (i_exp[i] >= e->n_experiment)
      e->n_experiment = i_exp[i] + 1;
    if (i_node[i] >= e->n_node)
      e->n_node = i_node[i] + 1;
  }
}

static int is_node_perturbed(experiment_t e, int i_node)
{
  int i;
  for (i = 0; i < e->n_perturbed; i++)
    if (e->perturbed[i] == i_node)
      return 1;
  return 0;
}

void experiment_set_write(FILE *f, const experiment_set_t e)
{
  int i_exp, i_node, i_state;
  for (i_exp = 0; i_exp < e->n_experiment; i_exp++) {
    const experiment_t ei = &e->experiment[i_exp];
    for (i_node = 0; i_node < e->n_node; i_node++)
      for (i_state = -1; i_state <= 1; i_state++)
	fprintf(f, "%d %d %c %f %d\n", i_exp, i_node, sym_from_state(i_state), score_for_state(ei,i_node,i_state), is_node_perturbed(ei, i_node));
  }
}

static void write_state(FILE *f, const int *state, int n_node)
{
  int i;
  for (i = 0; i < n_node; i++)
    fprintf(f, "%c", sym_from_state(state[i]));
}

static void write_repetition(FILE *f, const trajectory_t t)
{
  int i;
  for (i = 0; i <= t->repetition_end; i++) {
    fprintf(f, "%d: ", i);
    write_state(f, &t->state[i][0], t->n_node);
    fprintf(f, "\n");
  }
  fprintf(f, "s: ");
  write_state(f, t->steady_state, t->n_node);
  fprintf(f, "\n");
}

static void write_most_probable(FILE *f, const experiment_t e, int n_node)
{
  int i;
  for (i = 0; i < n_node; i++)
    fprintf(f, "%c", sym_from_state(most_probable_state(e,i)));
  fprintf(f, "\n");
}

double scale_factor(const experiment_set_t eset)
{
  return 1.0 / (eset->n_experiment * eset->n_node);
}

double lowest_possible_score(const experiment_set_t eset)
{
  double s = 0;
  int i;
  for (i = 0; i < eset->n_experiment; i++) {
    const experiment_t e = &eset->experiment[i];
    int j;
    for (j = 0; j < eset->n_node; j++)
      s += score_for_most_probable_state(e,j);
  }
  return s;
}

void network_advance_until_repetition(const network_t n, const experiment_t e, trajectory_t t)
{
  init_trajectory(t, e, n->n_node);
  int i;
  for (i = 1; i < MAX_STATES && !repetition_found(t); i++) {
    advance(n,t,i);
    check_for_repetition(t,i);
  }
}

void network_write_response_as_target_data(FILE *f, network_t n, const experiment_set_t e)
{
  const int n_node = n->n_node;
  if (n_node != e->n_node)
    die("network_write_response_from_experiment_set: network has %d nodes, experiment set has %d nodes",
	n_node, e->n_node);
  fprintf(f, "i_exp, i_node, outcome, value, is_perturbation\n");
  struct trajectory traj;
  int i_exp;
  for (i_exp = 0; i_exp < e->n_experiment; i_exp++) {
    network_advance_until_repetition(n, &e->experiment[i_exp], &traj);
    int i_node;
    for (i_node = 0; i_node < n_node; i_node++) {
      int i_outcome;
      for (i_outcome = -1; i_outcome <= 1; i_outcome++)
	fprintf(f, "%d, %d, %d, %.1f, %d\n",
		i_exp, i_node, i_outcome,
		fabs((double) traj.steady_state[i_node] - (double) i_outcome),
		traj.is_persistent[i_node] && traj.steady_state[i_node] == i_outcome);
    }
  }
}

void network_write_response_from_experiment_set(FILE *f, network_t n, const experiment_set_t e)
{
  const int n_node = n->n_node;
  if (n_node != e->n_node)
    die("network_write_response_from_experiment_set: network has %d nodes, experiment set has %d nodes",
	n_node, e->n_node);
  int i;
  struct trajectory traj;
  for (i = 0; i < e->n_experiment; i++) {
    fprintf(f, "experiment %d:\n", i);
    network_advance_until_repetition(n, &e->experiment[i], &traj);
    write_repetition(f,&traj);
    fprintf(f, "\n");
  }
  fprintf(f, "Lowest possible score: %g\n", lowest_possible_score(e));
  fprintf(f, "Most probable and predicted steady states:\n");
  for (i = 0; i < e->n_experiment; i++) {
    write_most_probable(f, &e->experiment[i], n_node);
    network_advance_until_repetition(n, &e->experiment[i], &traj);
    write_state(f, traj.steady_state, n_node);
    fprintf(f, "\n\n");
  }
}

static double score_for_trajectory(const experiment_t e, const trajectory_t t)
{
  int i_node;
  double s = 0;
  for (i_node = 0; i_node < t->n_node; i_node++) {
    const int si = t->steady_state[i_node];
    if (si == UNDEFINED)
      return LARGE_SCORE;
    s += score_for_state(e, i_node, si);
  }
  return s;
}

static double score(network_t n, const experiment_set_t eset, trajectory_t traj, double limit)
{
  double s = 0;
  int i_exp;
  for (i_exp = 0; i_exp < eset->n_experiment; i_exp++) {
    const experiment_t e = &eset->experiment[i_exp];
    network_advance_until_repetition(n, e, traj);
    if (repetition_found(traj)) {
      s += score_for_trajectory(e, traj); // * scale_factor(eset)
      if (s > limit)
	break;
    } else {
      return LARGE_SCORE;
    }
  }
  return s;
}

static void copy_network(network_t to, const network_t from)
{
  memcpy(&to->parent[0][0], &from->parent[0][0], to->n_node * to->n_parent * sizeof(int));
  memcpy(&to->outcome[0][0], &from->outcome[0][0], to->n_node * to->n_outcome * sizeof(int));
} 

#ifdef USE_MPI

static void send_int(int proc, int buf)
{
  MPI_Send(&buf, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
}

static int receive_int(int proc)
{
  int buf;
  MPI_Status s;
  MPI_Recv(&buf, 1, MPI_INT, proc, 0, MPI_COMM_WORLD, &s);
  return buf;
}

static void send_double(int proc, double buf)
{
  MPI_Send(&buf, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
}

static double receive_double(int proc)
{
  double buf;
  MPI_Status s;
  MPI_Recv(&buf, 1, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, &s);
  return buf;
}

static void exchange_networks(int proc, network_t n)
{
  MPI_Status s;
  MPI_Sendrecv_replace(&n->parent[0][0], n->n_node * n->n_parent, MPI_INT,
		       proc, 0, proc, 0, MPI_COMM_WORLD, &s);
  MPI_Sendrecv_replace(&n->outcome[0][0], n->n_node * n->n_outcome, MPI_INT,
		       proc, 0, proc, 0, MPI_COMM_WORLD, &s);
}

static void bcast_int(int *x)
{
  MPI_Bcast(x, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

#endif

static double fraction(unsigned long a, unsigned long b)
{
  if (b > 0)
    return (double) a / (double) b;
  return 0;
}

double network_monte_carlo(network_t n, 
			   const experiment_set_t e, 
			   unsigned long n_cycles,
			   unsigned long n_write,
			   double T_lo,
			   double T_hi,
			   FILE *out,
			   double target_score)
{
  const int n_node = n->n_node;
  double T = T_hi;
#ifdef USE_MPI
  int mpi_size = 0;
  int mpi_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  unsigned long exchange_acc = 0, exchange_tries = 0;
  T = T_lo * pow(T_hi/T_lo, fraction(mpi_rank, mpi_size-1));
#endif
  if (e->n_experiment == 0)
    die("network_monte_carlo: no experiments given");
  if (e->n_node < 2)
    die("network_monte_carlo: must have at least 2 nodes");
  if (n_node != e->n_node)
    die("network_monte_carlo: network has %d nodes, but experiment set has %d nodes", n_node, e->n_node);
  struct trajectory traj;
  double s = score(n,e,&traj,HUGE_VAL), s_best = s;
  fprintf(out, "number of steps: %lu\n", n_cycles);
  fprintf(out, "initial temperature: %g\n", T);
  fprintf(out, "target score: %g\n", target_score);
  fprintf(out, "initial score: %g\n", s);
  fprintf(out, "\n");
  fflush(out);
  struct network best;
  network_init(&best, n->n_node, n->n_parent);
  copy_network(&best, n);
  struct network t0;
  network_init(&t0, n->n_node, n->n_parent);
  unsigned long parent_acc = 0, parent_tries = 0, parent_moves = 1;
  unsigned long outcome_acc = 0, outcome_tries = 0, outcome_moves = 1;
  unsigned long i;
  for (i = 1; i <= n_cycles; i++) {
#ifndef USE_MPI
    /* if no MPI available, do annealing instead of replica exchange */
    T = T_hi * pow(T_lo/T_hi, fraction(i-1, n_cycles-1));
#endif
    copy_network(&t0, n);
    unsigned long j;
    const int is_parent_move = (i % 2) && n->n_parent < n->n_node - 1;
    if (is_parent_move) { /* change a parent */
      parent_tries++;
      for (j = 0; j < parent_moves; j++) {
	const int k = random_int_inclusive(0, n_node - 1); /* which node to change */
	int pnew;
      try_another_parent:
	pnew = random_int_inclusive(0, n_node - 1); /* new parent */
	if (pnew == k)
	  goto try_another_parent;
	int ip;
	for (ip = 0; ip < n->n_parent; ip++)
	  if (pnew == n->parent[k][ip])
	    goto try_another_parent;
	n->parent[k][random_int_inclusive(0, n->n_parent - 1)] = pnew;
	qsort(&n->parent[k][0], n->n_parent, sizeof(int), intcmp);
      }
    } else { /* change outcomes */
      outcome_tries++;
      const int i_all_parents_unperturbed = (n->n_outcome - 1)/2;
      for (j = 0; j < outcome_moves; j++) {
	const int k = random_int_inclusive(0, n_node - 1);
	/* change outcomes */
	if (n->n_parent > 0) {
	  int i_outcome;
	try_another_outcome:
	  i_outcome = random_int_inclusive(0, n->n_outcome - 1);
	  if (i_outcome == i_all_parents_unperturbed)
	    goto try_another_outcome;
	  n->outcome[k][i_outcome] = random_int_inclusive(-1,1);
	}
      }
    }
    const double limit = s - T*log(uniform_random_from_0_to_1_exclusive());
    const double s_new = score(n, e, &traj, limit);
    if (s_new < 0.9*LARGE_SCORE && s_new < limit) { 
      /* accepted */
      if (is_parent_move)
	parent_acc++;
      else
	outcome_acc++;
      s = s_new;
      if (s < s_best) {
	s_best = s;
	copy_network(&best, n);
      }
    } else {
      /* rejected */
      copy_network(n, &t0);
    }
#ifdef USE_MPI
    const int try_exchange = i % EXCHANGE_INTERVAL == 0;
    if (try_exchange) {
      if ((mpi_rank + i/EXCHANGE_INTERVAL) % 2) {
	if (mpi_rank < mpi_size - 1) {
	  /* Try an exchange with higher-T neighbor */
	  const int r1 = mpi_rank + 1;
	  const double T1 = receive_double(r1);
	  const double s1 = receive_double(r1);
	  const double ds = s1 - s;
	  if (ds < 0 || (T > 0 && uniform_random_from_0_to_1_exclusive() < exp(-ds*(1/T - 1/T1)))) {
	    send_int(r1, 1); /* accepted - tell neighbor */
	    exchange_networks(r1,n);
	    s = s1;
	    if (s < s_best) {
	      s_best = s;
	      copy_network(&best, n);
	    }
	  } else {
	    send_int(r1, 0); /* rejected - tell neighbor */
	  }
	}
      } else {
	if (mpi_rank > 0) {
	  /* lower-T neighbor is trying an exchange with us */
	  exchange_tries++;
	  const int r0 = mpi_rank - 1;
	  send_double(r0, T);
	  send_double(r0, s);
	  if (receive_int(r0)) { /* neighbor says the exchange was accepted */
	    exchange_networks(r0,n);
	    exchange_acc++;
	  }
	}
      }
    } /* try_exchange */
#endif
    int stop = 0;
#ifdef USE_MPI
    if (try_exchange) {
      if (mpi_rank == 0 && s_best <= target_score)
	stop = 1;
      bcast_int(&stop);
    }
#else
    if (s_best <= target_score)
      stop = 1;
#endif
    if (out && (stop || (n_write > 0 && n_cycles > n_write && i % (n_cycles/n_write) == 0))) {
      fprintf(out, "Ran %lu steps.\n", i);
      fprintf(out, "Temperature: %g\n", T);
      fprintf(out, "Parent move acceptances since last adjust: %lu\n", parent_acc);
      fprintf(out, "Parent move tries since last adjust: %lu\n", parent_tries);
      fprintf(out, "Fraction of parent moves accepted since last adjust: %g\n", fraction(parent_acc, parent_tries));
      fprintf(out, "Outcome move acceptances since last adjust: %lu\n", outcome_acc);
      fprintf(out, "Outcome move tries since last adjust: %lu\n", outcome_tries);
      fprintf(out, "Fraction of outcome moves accepted since last adjust: %g\n", fraction(outcome_acc, outcome_tries));
      fprintf(out, "Number of parent moves per cycle: %lu\n", parent_moves);
      fprintf(out, "Number of outcome moves per cycle: %lu\n", outcome_moves);
#ifdef USE_MPI
      fprintf(out, "Process %d of %d\n", mpi_rank, mpi_size);
      if (exchange_tries > 0)
	fprintf(out, "Fraction of exchanges with lower-T neighbor accepted: %g\n", fraction(exchange_acc, exchange_tries));
#endif
      fprintf(out, "Best score: %g\n", s_best);
      fprintf(out, "\n");
      fflush(out);
    }
    if (stop)
      break;
    /* adjust number of moves */
    if (parent_tries == ADJUST_MOVE_SIZE_INTERVAL) { 
      const double f = fraction(parent_acc, parent_tries);
      if (f > 0.5 && parent_moves < n_node)
	parent_moves++;
      else if (f < 0.5 && parent_moves > 1)
	parent_moves--;
      parent_tries = 0;
      parent_acc = 0;
    }
    if (outcome_tries == ADJUST_MOVE_SIZE_INTERVAL) {
      const double f = fraction(outcome_acc, outcome_tries);
      if (f > 0.5)
	outcome_moves++;
      else if (f < 0.5 && outcome_moves > 1)
	outcome_moves--;
      outcome_tries = 0;
      outcome_acc = 0;
    }
  }
  copy_network(n, &best);
  network_delete(&best);
  network_delete(&t0);
  return s_best;
}
