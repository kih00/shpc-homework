#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fallback_print_usage() {
  printf("Usage: ./convert [int|long|float|double] number\n");
  printf("Example: ./convert float 3.14\n");
  exit(0);
}

void print_int(int x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  union {
    int x;
    unsigned int b;
  } tp;
  tp.x = x;
  
  for (int i=0; i<32; i++) {
    output[31-i] = (tp.b & 1) + '0';
    tp.b >>= 1;
  }
  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_long(long x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  union {
    long x;
    unsigned long b;
  } tp;
  tp.x = x;

  for (int i=0; i<64; i++) {
    output[63-i] = (tp.b & 1) + '0';
    tp.b >>= 1;
  }
  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_float(float x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };
  /* YOUR CODE START HERE */
  union {
    float x;
    unsigned int b;
  } tp;
  tp.x = x;
  
  for (int i=0; i<32; i++) {
    output[31-i] = (tp.b & 1) + '0';
    tp.b >>= 1;
  }
  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_double(double x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  union {
    double x;
    unsigned long b;
  } tp;
  tp.x = x;
  
  for (int i=0; i<64; i++) {
    output[63-i] = (tp.b & 1) + '0';
    tp.b >>= 1;
  }
  /* YOUR CODE END HERE */
  printf("%s\n", output);
}

int main(int argc, char **argv) {
  if (argc != 3)
    fallback_print_usage();
  if (strcmp(argv[1], "int") == 0) {
    print_int(atoi(argv[2]));
  } else if (strcmp(argv[1], "long") == 0) {
    print_long(atol(argv[2]));
  } else if (strcmp(argv[1], "float") == 0) {
    print_float(atof(argv[2]));
  } else if (strcmp(argv[1], "double") == 0) {
    print_double(atof(argv[2]));
  } else {
    fallback_print_usage();
  }
}
