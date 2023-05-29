//
// Created by Zhou on 2023/4/10.
//

#ifndef KD_TREE_KD_TREE_H
#define KD_TREE_KD_TREE_H

#include "kernel.h"

class KdTree
{
private:
    int k;          // k-tree such as 2-tree, 3-tree
    int m;          // target points number, equal n or less n;
    int n;          // reference points number
    int f;          // function number
    float *referencePoints{};             // query scope points
    float *searchPoints{};                // target points

public:
    KdTree(int k, int m, int n, int f) {
        if((k < 0) || (m < 0) || (n < 0)) {
            printf("k, m, n must be positive!\n");
            exit(1);
        }

        // check f in [0,8]
        if((f < 0) || (f > 8)) {
            printf("f must be in [0,8]!\n");
            exit(1);
        }

        this->k = k;
        this->m = m;
        this->n = n;
        this->f = f;
    }

    ~KdTree() {
        free(referencePoints);
        free(searchPoints);
    };

    void build(const float *buffer);

    void search(const float *buffer, int **results);
};

void KdTree::build(const float *buffer) {
    auto *tmp = (float*)malloc(sizeof(float) * k * n);
    assert(tmp != nullptr);
    // x y in default
    for (int i = 0; i < k * n; i++) {
        tmp[i] = buffer[i];
    }
    referencePoints = tmp;
}

void KdTree::search(const float *buffer, int **results) {
    auto *tmp = (float*)malloc(sizeof(float) * k * m);
    assert(tmp != nullptr);
    // x y in default
    for (int i = 0; i < k * m; i++) {
        tmp[i] = buffer[i];
    }
    searchPoints = tmp;

    // cudaCallback depends on f
    switch (f) {
        case 0:
            v0::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        case 1:
            v1::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        case 2:
            v2::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        case 3:
            v3::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        case 4:
            v4::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        case 5:
            v5::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        case 6:
            v6::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        case 7:
            v7::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        case 8:
            v8::cudaCallback(k, m, n, searchPoints, referencePoints, results);
            break;
        default:
            printf("f must be in [0,8]!\n");
            exit(1);
    }
}


#endif //KD_TREE_KD_TREE_H
