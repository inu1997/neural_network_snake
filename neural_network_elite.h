#ifndef __NEURAL_NETWORK_ELITE_H
#define __NEURAL_NETWORK_ELITE_H

#include "neural_network.h"

typedef struct {
	int max_len;
	void *list_head;
} NNEliteList;

void nn_elites_init_list(NNEliteList *list, int max_len);

void nn_elites_add(NNEliteList *list, NeuralNetwork *nn, float goodness);

void nn_elites_clear(NNEliteList *list);

NeuralNetwork *nn_elites_pick_by_random(NNEliteList *list, NeuralNetwork *dont_pick);

NeuralNetwork *nn_elites_get_best(NNEliteList *list);

int nn_elites_get_count(NNEliteList *list);

int nn_elites_save(NNEliteList *list, const char *file_name);

int nn_elites_load(NNEliteList *list, const char *file_name);

int nn_elites_savef(NNEliteList *list, FILE *f);

int nn_elites_loadf(NNEliteList *list, FILE *f);

void nn_elite_show(NNEliteList *list);

#endif /* __NEURAL_NETWORK_ELITE_H */
