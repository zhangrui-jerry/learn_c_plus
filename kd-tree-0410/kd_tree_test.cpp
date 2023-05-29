// kd tree class test
//
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include "kd_tree.h"

void random_init_data(int k, int m, int n, float **searchPoints,
                      float **referencePoints);

void show_points(float *data, int n, int k);

int main()
{
    printf("KD_tree in GPU !\n");

    int k = 3;
    int m = 4;
    int n = 1024;

    float *searchPoints;
    float *referencePoints;

    random_init_data(k, m, n, &searchPoints, &referencePoints);

//    printf("searchPoints: \n");
//    show_points(searchPoints, m, k);
//    printf("referencePoints: \n");
//    show_points(referencePoints, n, k);

    // use for assert result
    int **baselineResults = NULL;

    int numSamples = 0;
    baselineResults = (int **)malloc(sizeof(int *) * numSamples);
    for (int i = 0; i < numSamples; i++) {
        baselineResults[i] = NULL;
    }

    int *results;

    for(int i = 0; i < numSamples; i++)
    {
        auto *kdTree = new KdTree(k, m, n, i);

        kdTree->build(referencePoints);

        kdTree->search(searchPoints, &results);

        baselineResults[i] = results;
    }

    // show results
    for(int i = 0; i < numSamples; i++)
    {
        printf("results[%d]: ", i);
        for(int j = 0; j < m; j++)
        {
            printf("%d ", baselineResults[i][j]);
        }
        printf("\n");
    }

    return 0;
}

void random_init_data(int k, int m, int n, float **searchPoints,
                      float **referencePoints) {
    srand(1000);
    float *tmp;

    tmp = (float*)malloc(sizeof(float) * k * m);
    assert(tmp != NULL);
    for (int i = 0; i < k * m; i++) {
        tmp[i] = rand() / (double)RAND_MAX;
    }
    *searchPoints = tmp;

    tmp = (float*)malloc(sizeof(float) * k * n);
    assert(tmp != NULL);
    for (int i = 0; i < k * n; i++) {
        tmp[i] =  rand() / (double)RAND_MAX;
    }
    *referencePoints = tmp;
}

void show_points(float *data, int n, int k)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < k; j++)
        {
            printf("%f ", data[i * k + j]);
        }
        printf("\n");
    }
}
