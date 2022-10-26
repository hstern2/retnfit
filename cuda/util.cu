#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdarg.h>
#include <ctype.h>
#include <math.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "util.h"

void die(const char *fmt, ...)
{
  va_list argp;
  va_start(argp, fmt);
  char *ret;
  if (vasprintf(&ret, fmt, argp) == -1) {
    fprintf(stderr, "vasprintf failed\n");
    return;
  }
  va_end(argp);
  char buf[1024];
  sprintf(buf, "%s\n", ret);
  free(ret);
  fprintf(stderr, buf);
}

void *safe_malloc(size_t size)
{
  void *p = malloc(size);
  if (!p)
    die("safe_malloc: could not allocate %lu bytes", size);
  return p;
}

FILE *safe_fopen(const char *path, const char *mode)
{
  FILE *f = fopen(path,mode);
  if (!f)
    die("safe_fopen: could not open file \'%s\'", path);
  return f;
}

int end_of_file(FILE *f)
{
  int c = fgetc(f);
  if (c == EOF)
    return 1;
  ungetc(c, f);
  return 0;
}

int isdigits(const char *s)
{
  for ( ; *s; s++)
    if (!isdigit(*s))
      return 0;
  return 1;
}

int string_begins_with(const char *buf, const char *start)
{
  return !strncmp(buf, start, strlen(start));
}

int intcmp(const void *a, const void *b)
{
  if (*(const int *) a < *(const int *) b)
    return -1;
  if (*(const int *) a > *(const int *) b)
    return 1;
  return 0;
}

void read_line(FILE *f, char *buf, int n)
{
  if (!fgets(buf, n, f))
    die("read_line: unexpected end of file");
  if (strlen(buf) >= n)
    die("read_line: line too long");
}

double uniform_random_from_0_to_1_exclusive()
{
  return (double) random() / ((double) RAND_MAX + 1.0);
  /* return unif_rand(); */
}

int random_int_inclusive(int a, int b)
{
  return (int) floor((b-a+1)*uniform_random_from_0_to_1_exclusive()) + a;
}
