/* $Id: array.c,v 1.5 2011/05/11 22:17:17 hstern Exp $ */

/*
 * Copyright (c) 2008 Harry A. Stern
 * Copyright (c) 2008 University of Rochester
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */


#include <stdlib.h>
#include "array.h"

ARRAY_DEFINITION(int)

#ifdef SIMPLE_EXAMPLE

int main()
{
  int i, j, k, n, **a, ***b;
  double2_t ***c;
  /* 2D array */
  a = int_array2D_new(4, 2);
  n = 0;
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      a[i][j] = n++;
  for (i = 0; i < 4; i++) {
    for (j = 0; j < 2; j++)
      printf("%4d", a[i][j]);
    printf("\n");
  }
  printf("\n\n");
  int_array2D_delete(a);
  /* 3D array of int */
  b = int_array3D_new(4, 2, 8);
  n = 0;
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      for (k = 0; k < 8; k++)
	b[i][j][k] = n++;
  for (i = 0; i < 4; i++) {
    for (j = 0; j < 2; j++) {
      for (k = 0; k < 8; k++)
	printf("%4d", b[i][j][k]);
      printf("\n");
    }
    printf("\n");
  }
  int_array3D_delete(b);
  /* 3D array of double2_t */
  c = double2_t_array3D_new(4, 2, 8);
  n = 0;
  for (i = 0; i < 4; i++)
    for (j = 0; j < 2; j++)
      for (k = 0; k < 8; k++) {
	c[i][j][k][0] = n * 0.1;
	c[i][j][k][1] = n * 0.2;
	n++;
      }
  for (i = 0; i < 4; i++) {
    for (j = 0; j < 2; j++) {
      for (k = 0; k < 8; k++)
	printf("%.1f %.1f  ", c[i][j][k][0], c[i][j][k][1]);
      printf("\n");
    }
    printf("\n");
  }
  double2_t_array3D_delete(c);
}

#endif
