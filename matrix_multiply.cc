#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include "carbon_user.h"
#include <iostream>
using namespace std;
//#define DEBUG 1

class ThreadData
{
public:
   ThreadData(int tid_, int num_threads_, int m_, int n_, int p_, float** matA_, float** matB_, float** matC_)
      : tid(tid_), num_threads(num_threads_)
      , m(m_), n(n_), p(p_)
      , matA(matA_), matB(matB_), matC(matC_)
   {}
   ~ThreadData()
   {}

   int tid;
   int num_threads;
   int m;
   int n;
   int p;
   float** matA;
   float** matB;
   float** matC;
};

void* threadMain(void* arg);
void verifyOutputs(int m, int n, int p, float** matA, float** matB, float** matC);
void printMatrix(float** mat, int num_rows, int num_cols);
void printUsage();

int main(int argc, char *argv[])
{
   if (argc != 6)
   {
      printUsage();
      exit(EXIT_FAILURE);
   }

   int m = atoi(argv[1]);
   int n = atoi(argv[2]);
   int p = atoi(argv[3]);
   int num_threads = atoi(argv[4]);
   bool verify_outputs = (atoi(argv[5]) != 0);

   printf("\nMatrix Multiply: M(%i), N(%i), P(%i), Num Threads(%i), Verify(%s) Start.\n",
          m, n, p, num_threads, verify_outputs ? "YES" : "NO");

   if (p % num_threads != 0)
   {
      printUsage();
      exit(EXIT_FAILURE);
   }

   // Initialize & allocate matrices (A, B, C)
   float** matA;
   float** matB;
   float** matC;

   posix_memalign((void**) &matA, 64, m * sizeof(float*)); 
   for (int i = 0; i < m; i++)
   {
      posix_memalign((void**) &matA[i], 64, n * sizeof(float));
      for (int j = 0; j < n; j++)
         matA[i][j] = i*j;
   }

   posix_memalign((void**) &matB, 64, n * sizeof(float*)); 
   for (int i = 0; i < n; i++)
   {
      posix_memalign((void**) &matB[i], 64, p * sizeof(float));
      for (int j = 0; j < p; j++)
         matB[i][j] = i+j;
   }

   posix_memalign((void**) &matC, 64, m * sizeof(float*)); 
   for (int i = 0; i < m; i++)
   {
      posix_memalign((void**) &matC[i], 64, p * sizeof(float));
      for (int j = 0; j < p; j++)
         matC[i][j] = 0.0;
   }


#ifdef DEBUG
   printf("A = \n");
   printMatrix(matA, m, n);
   printf("B = \n");
   printMatrix(matB, n, p);
#endif

   ThreadData* thread_args[num_threads];
   for (int i = 0; i < num_threads; i++)
   {
      thread_args[i] = new ThreadData(i, num_threads, m, n, p, matA, matB, matC);
   }

#ifdef DEBUG
   fprintf(stderr, "Created Thread Args.\n");;
#endif

   // Enable performance and energy models
   CarbonEnableModels();

   pthread_t thread_handles[num_threads];
   for (int i = 1; i < num_threads; i++)
   {
      int ret = pthread_create(&thread_handles[i], NULL, threadMain, (void*) thread_args[i]);
      if (ret != 0)
      {
         fprintf(stderr, "ERROR spawning thread %i\n", i);
         exit(EXIT_FAILURE);
      }
   }
   threadMain((void*) thread_args[0]);

#ifdef DEBUG
   fprintf(stderr, "Created Threads.\n");
#endif

   for (int i = 1; i < num_threads; i++)
   {
      pthread_join(thread_handles[i], NULL);
   }

   // Disable performance and energy models
   CarbonDisableModels();

#ifdef DEBUG
   printf("C = \n");
   printMatrix(matC, m, p);
#endif
   
   printf("Matrix Multiply: Done.\n");
   
   if (verify_outputs)
      verifyOutputs(m, n, p, matA, matB, matC);

   // Free matrices (A, B, C)
   for (int i = 0; i < m; i++)
     free(matA[i]);
   free(matA); 
   for (int i = 0; i < n; i++)
     free(matB[i]);
   free(matB); 
   for (int i = 0; i < m; i++)
     free(matC[i]);
   free(matC);

   return 0;
}

void* threadMain(void* arg)
{
   ThreadData* thread_arg = (ThreadData*) arg;

   int tid           = thread_arg->tid;
   int num_threads   = thread_arg->num_threads;
   int m             = thread_arg->m;
   int n             = thread_arg->n;
   int p             = thread_arg->p;
   float** matA      = thread_arg->matA;
   float** matB      = thread_arg->matB;
   float** matC      = thread_arg->matC;
   
   int p_min = 0;
   int p_max = 0;
   int partition = 4;
   int tile_width = p/partition;

#ifdef DEBUG
   fprintf(stderr, "Thread: Tid(%i), Num Threads(%i), m(%i), n(%i), p(%i), p_min(%i), p_max(%i)\n",
           tid, num_threads, m, n, p, p_min, p_max);
#endif

   for(int a = 0; a < partition; a++)
   {
      for(int b = 0; b < partition; b++)
      {
       
         p_min = tid * (tile_width / num_threads)+ b*tile_width;
         p_max = (tid+1) * (tile_width / num_threads) + b*tile_width;

         for(int c = 0; c < partition; c++)
         {
             for(int i = a*tile_width; i < (a*tile_width)+tile_width ; i++) 
            {
               for(int j = p_min; j < p_max; j++)
               {
                  for(int k = c*tile_width; k < (c*tile_width)+tile_width; k++)
                  {
                     matC[i][j] +=  matA[i][k] * matB[k][j];
                  }
               }
             }
         }
      }
   }

/* Sequential version

   for (int i = 0; i < m; i++)
   {
      for (int j = p_min; j < p_max; j++)
      {
         matC[i][j] = 0.0;
         for (int k = 0; k < n; k++)
         {
            matC[i][j] += matA[i][k] * matB[k][j];
         }
      }
   } */

   return 0;
}

void verifyOutputs(int m, int n, int p, float** matA, float** matB, float** matC)
{
   for (int i = 0; i < m; i++)
   {
      for (int j = 0; j < p; j++)
      {
         float act_val = 0.0;
         for (int k = 0; k < n; k++)
         {
            act_val += (matA[i][k] * matB[k][j]);
         }
         if (act_val != matC[i][j])
         {
            fprintf(stderr, "Verification Failure: Row-ID(%i), Column-ID(%i), Calculated(%f), Actual(%f).",
                    i, j, matC[i][j], act_val);
            exit(EXIT_FAILURE);
         }
      }
   }
   printf("Matrix Multiply: Verification Successful.\n");
}

void printMatrix(float** mat, int num_rows, int num_cols)
{
   for (int i = 0; i < num_rows; i++)
   {
      for (int j = 0; j < num_cols; j++)
         printf("%f\t", mat[i][j]);
      printf("\n");
   }
}

void printUsage()
{
   fprintf(stderr, "\n[Usage]: ./matrix_mulitply <M> <N> <P> <NT> <VERIFY>\n");
   fprintf(stderr, "M, N, P - Matrix Dimensions\n");
   fprintf(stderr, "VERIFY - Verify Outputs [0 to NOT verify, > 0 to verify]\n");
   fprintf(stderr, "NT - Number of Threads\n");
   fprintf(stderr, "P should be a multiple of NT\n");
}
