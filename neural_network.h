#ifndef __NEURAL_NETWORK_H
#define __NEURAL_NETWORK_H

#include <stdio.h>

typedef enum {
	ACT_FUNC_TYPE_LINEAR,
	ACT_FUNC_TYPE_SIGMOID,
	ACT_FUNC_TYPE_TANH,
} ACT_FUNC_TYPE;

typedef struct {
	int n_input;
	int n_output;
	int n_hidden;
	int n_neuro_per_hidden;
	int use_bias;
	ACT_FUNC_TYPE act_func_type_hidden;
	ACT_FUNC_TYPE act_func_type_output;

	/* A cache to get the number of neuro and weight */
	int _n_neuro;
	int _n_weight;

	float *weight;
	float *bias;
	float *output;
	float *delta;
} NeuralNetwork;

NeuralNetwork *nn_create(int n_input,
		int n_output,
		int n_hidden,
		int n_neuro_per_hidden,
		int use_bias,
		ACT_FUNC_TYPE act_func_type_hidden,
		ACT_FUNC_TYPE act_func_type_output);

NeuralNetwork *nn_produce(NeuralNetwork *a, NeuralNetwork *b);

void nn_free(NeuralNetwork *nn);

NeuralNetwork *nn_duplicate(NeuralNetwork *nn);

float *nn_run(NeuralNetwork *nn, float *input);

float *nn_train(NeuralNetwork *nn, float *input, float *expect, float rate);

void nn_plus_randomize(NeuralNetwork *nn, float range);

void nn_plus_randomize_by_rate(NeuralNetwork *nn, float range, float rate);

void nn_randomize(NeuralNetwork *nn);

void nn_randomize_with_scale(NeuralNetwork *nn, float scale);

void nn_randomize_by_rate(NeuralNetwork *nn, float rate);

void nn_randomize_with_scale_by_rate(NeuralNetwork *nn, float scale, float rate);

int nn_save(NeuralNetwork *nn, const char * file_name);

NeuralNetwork *nn_load(const char *file_name);

int nn_savef(NeuralNetwork *nn, FILE *f);

NeuralNetwork *nn_loadf(FILE *f);

#endif /* __NEURAL_NETWORK_H */
