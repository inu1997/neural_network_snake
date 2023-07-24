#include "neural_network.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int random_pick(float rate);

static float nn_gen_random();

static float nn_gen_random_zero_to_one();

static int nn_compute_n_weight(NeuralNetwork *nn);

static void nn_forward_propagation(ACT_FUNC_TYPE act_func_type,
		int use_bias,
		float *input,
		int n_input,
		float *output,
		int n_output,
		float *bias,
		float *weight);

static void nn_correct(float *weight, float *delta, float *input, int n_input, int n_output, float rate);

static float nn_act_func_derivate(ACT_FUNC_TYPE act_func_type, float output);

static int
random_pick(float rate)
{
	if (nn_gen_random_zero_to_one() < rate)
		return 1;
	return 0;
}

static float
nn_gen_random()
{
	return nn_gen_random_zero_to_one() - 0.5f;	/* A random -0.5 ~ 0.5 */
}

static float nn_gen_random_zero_to_one()
{
	float r;

	r = rand();			 /* A random 0 ~ RAND_MAX */
	r /= (float)RAND_MAX;   /* A random 0 ~ 1.0 */

	return r;
}

static int
nn_compute_n_weight(NeuralNetwork *nn)
{
	int n_weight;
	int i;
	int n_input;

	n_input = nn->n_input;
	n_weight = 0;
	for (i = 0; i < nn->n_hidden; i++)
	{
		n_weight += n_input * nn->n_neuro_per_hidden;
		n_input = nn->n_neuro_per_hidden;
	}

	n_weight += n_input * nn->n_output;

	return n_weight;
}

static void
nn_forward_propagation(ACT_FUNC_TYPE act_func_type,
		int use_bias,
		float *input,
		int n_input,
		float *output,
		int n_output,
		float *bias,
		float *weight)
{
	int i;
	int j;

	for (i = 0; i < n_output; i++)
	{
		if (use_bias)
			output[i] = bias[i];
		else
			output[i] = 0;
		/* w vector dot i vector + bias */
		for (j = 0; j < n_input; j++)
		{
			output[i] += weight[i * n_input + j] * input[j];
		}
		/* Do activation function */
		switch (act_func_type)
		{
			case ACT_FUNC_TYPE_SIGMOID:
				output[i] = 1.0f / (1.0f + exp(-output[i]));
				break;
			case ACT_FUNC_TYPE_TANH:
				output[i] = tanh(output[i]);
				break;
			default:
				break;
		}
	}
}

static void
nn_correct(float *weight, float *delta, float *input, int n_input, int n_output, float rate)
{
	int i;
	int j;
	for (i = 0; i < n_output; i++)
	{
		for (j = 0; j < n_input; j++)
		{
			weight[i * n_input + j] += delta[i] * input[j] * rate;
		}
	}
}

static float
nn_act_func_derivate(ACT_FUNC_TYPE act_func_type, float output)
{
	switch (act_func_type)
	{
		case ACT_FUNC_TYPE_SIGMOID:
			return output * (1 - output);

		case ACT_FUNC_TYPE_TANH:
			return 1 - output * output;

		default:
			break;
	}
	return 1.0f;
}

NeuralNetwork *
nn_create(int n_input,
		int n_output,
		int n_hidden,
		int n_neuro_per_hidden,
		int use_bias,
		ACT_FUNC_TYPE act_func_type_hidden,
		ACT_FUNC_TYPE act_func_type_output)
{
	int i;
	NeuralNetwork *nn;

	/* Error check */
	if (n_input < 0)
		return NULL;
	if (n_output < 0)
		return NULL;
	if (n_hidden < 0)
		return NULL;
	if (n_hidden > 0 && n_neuro_per_hidden < 1)
		return NULL;

	nn = malloc(sizeof(*nn));
	nn->n_input = n_input;
	nn->n_output = n_output;
	nn->n_hidden = n_hidden;
	nn->n_neuro_per_hidden = n_neuro_per_hidden;
	nn->use_bias = use_bias;
	nn->act_func_type_hidden = act_func_type_hidden;
	nn->act_func_type_output = act_func_type_output;
	/* Calculate number of neuro */
	nn->_n_neuro = n_output + n_hidden  * n_neuro_per_hidden;
	nn->_n_weight = nn_compute_n_weight(nn);

	nn->weight = malloc(nn->_n_weight * sizeof(float));
	if (nn->use_bias)
		nn->bias = malloc(nn->_n_neuro * sizeof(float));
	nn->output = malloc(nn->_n_neuro * sizeof(float));
	nn->delta = malloc(nn->_n_neuro * sizeof(float));

	nn_randomize(nn);

	return nn;
}

NeuralNetwork *
nn_produce(NeuralNetwork *a, NeuralNetwork *b)
{
	int i;
	NeuralNetwork *nn;

	if (a->n_input != b->n_input)
		return NULL;
	if (a->n_output != b->n_output)
		return NULL;
	if (a->n_hidden != b->n_hidden)
		return NULL;
	if (a->n_neuro_per_hidden != b->n_neuro_per_hidden)
		return NULL;
	if (a->act_func_type_hidden != b->act_func_type_hidden)
		return NULL;
	if (a->act_func_type_output != b->act_func_type_output)
		return NULL;

	nn = nn_create(a->n_input,
			a->n_output,
			a->n_hidden,
			a->n_neuro_per_hidden,
			a->use_bias,
			a->act_func_type_hidden,
			a->act_func_type_output);

	if (nn->use_bias)
	{
		for (i = 0; i < a->_n_neuro; i++)
		{
			nn->bias[i] = rand() & 1 ? a->bias[i] : b->bias[i];
		}
	}

	for (i = 0; i < a->_n_weight; i++)
	{
		nn->weight[i] = rand() & 1 ? a->weight[i] : b->weight[i];
	}

	return nn;
}

void
nn_free(NeuralNetwork *nn)
{
	free(nn->weight);
	if (nn->use_bias)
		free(nn->bias);
	free(nn->output);
	free(nn->delta);
	free(nn);
}

NeuralNetwork *
nn_duplicate(NeuralNetwork *nn)
{
	NeuralNetwork *new_nn;

	if (nn == NULL)
		return NULL;

	new_nn = nn_create(nn->n_input,
			nn->n_output,
			nn->n_hidden,
			nn->n_neuro_per_hidden,
			nn->use_bias,
			nn->act_func_type_hidden,
			nn->act_func_type_output);

	memcpy(new_nn->weight, nn->weight, nn->_n_weight * sizeof(float));
	if (nn->use_bias)
		memcpy(new_nn->bias, nn->bias, nn->_n_neuro * sizeof(float));

	return new_nn;
}

float *
nn_run(NeuralNetwork *nn, float *input)
{
	int i;
	int j;
	int k;
	float *output;  /* Output buffer of this layer */
	float *bias;	/* Bias of this layer */
	float *weight;  /* Weight matrix of this layer */
	int n_input;	/* Number of input or Number of output of previous layer */
	int n_output;   /* Number of output of this layer */

	n_input = nn->n_input;
	output = nn->output;
	if (nn->use_bias)
		bias = nn->bias;
	weight = nn->weight;
	/*
	 * 1. Process the hidden layers if any
	 */
	for (i = 0; i < nn->n_hidden; i++)
	{
		/* So many outputs this layer */
		n_output = nn->n_neuro_per_hidden;
		/* Forward propergation */
		nn_forward_propagation(nn->act_func_type_hidden,
				nn->use_bias,
				input,
				n_input,
				output,
				n_output,
				bias,
				weight);

		/* Move pointer forward to the next layer */
		input = output; /* Output of this layer is the next layer's input */
		output += n_output;			 /* Forwrad to the next layer */
		if (nn->use_bias)
			bias += n_output;
		weight += n_input * n_output;   /* Forward to the next layer */
		/* Set the number of input to the previous layer */
		n_input = nn->n_neuro_per_hidden;
	}

	/*
	 * 2. Process the output layer.
	 */
	/* So many outputs this layer */
	n_output = nn->n_output;
	/* Forward propergation */
	nn_forward_propagation(nn->act_func_type_output,
			nn->use_bias,
			input,
			n_input,
			output,
			n_output,
			bias,
			weight);

	return output;
}

float *
nn_train(NeuralNetwork *nn, float *input, float *expect, float rate)
{
	int i;
	int j;
	int k;
	float *ret;
	int n_output;	   /* Number of output of this layer */
	int n_next_output;   /* Number of the neuro of next layer */
	float *delta;	   /* Delta of this layer */
	float *output;	  /* Output of this layer */
	float *bias;		/* Bias of this layer */
	float *next_delta;  /* delta of next layer */
	float *next_weight; /* delta of next layer */

	/*
	 * 0. Run once
	 */
	ret = nn_run(nn, input);

	/*
	 * 1. From the output layer, do back propagation computation.
	 */
	n_output = nn->n_output;
	output = &nn->output[nn->_n_neuro - nn->n_output];
	if (nn->use_bias)
		bias = &nn->bias[nn->_n_neuro - nn->n_output];
	delta = &nn->delta[nn->_n_neuro - nn->n_output];

	/*
	 * Compute delta of this layer, also fix bias of this layer
	 */
	for (i = 0; i < n_output; i++)
	{
		delta[i] = expect[i] - output[i];

		/* Apply derivation of activation function of this neuro */
		delta[i] *= nn_act_func_derivate(nn->act_func_type_output, output[i]);

		if (nn->use_bias)
			bias[i] += delta[i] * rate;
	}

	/*
	 * 2. From the last hidden layer, do back propagation computation
	 */
	next_weight = &nn->weight[nn->_n_weight];
	for (i = 0; i < nn->n_hidden; i++)
	{
		n_next_output = n_output;
		n_output = nn->n_neuro_per_hidden;
		/* Move weight to this layer */
		next_weight -= n_next_output * n_output;

		/* Move next_delta, delta, output to this layer */
		next_delta = delta;
		delta -= n_output;
		if (nn->use_bias)
			bias -= n_output;
		output -= n_output;

		/*
		 * a. Compute delta of this layer, also fix bias of this layer
		 */
		for (j = 0; j < n_output; j++)
		{
			/*
			 * The j-th neuro's delta is
			 * "the next layer's delta" dot t"he j-th column vector of the next layer's weight matrix"
			 * times the derivation of this neuro
			 */
			delta[j] = 0;
			for (k = 0; k < n_next_output; k++)
			{
				delta[j] += next_delta[k] * next_weight[k * n_output + j];
			}

			/* Apply derivation of this neuro */
			delta[j] *= nn_act_func_derivate(nn->act_func_type_hidden, output[j]);

			if (nn->use_bias)
				bias[j] += delta[j] * rate;
		}

		/*
		 * b. Correct the next layer's weight
		 */
		nn_correct(next_weight, next_delta, output, n_output, n_next_output, rate);
	}

	n_next_output = n_output;
	n_output = nn->n_input;
	/* Move weight to this layer */
	next_weight -= n_next_output * n_output;

	/* Move next_delta, output to this layer */
	next_delta = delta;
	output = input; /* Input is treated as the output of this "input layer" */

	/*
	 * Correct the next layer's weight
	 */
	nn_correct(next_weight, next_delta, output, n_output, n_next_output, rate);
	return ret;
}

void
nn_plus_randomize(NeuralNetwork *nn, float range)
{
	int i;

	if (nn->use_bias)
	{
		for (i = 0; i < nn->_n_neuro; i++)
		{
			nn->bias[i] += nn_gen_random() * 2 * range;
		}
	}

	for (i = 0; i < nn->_n_weight; i++)
	{
		nn->weight[i] += nn_gen_random() * 2 * range;
	}
}

void
nn_plus_randomize_by_rate(NeuralNetwork *nn, float range, float rate)
{
	int i;

	if (nn->use_bias)
	{
		for (i = 0; i < nn->_n_neuro; i++)
		{
			if (random_pick(rate))
				nn->bias[i] += nn_gen_random() * 2 * range;
		}
	}

	for (i = 0; i < nn->_n_weight; i++)
	{
		if (random_pick(rate))
			nn->weight[i] += nn_gen_random() * 2 * range;
	}

}

void
nn_randomize(NeuralNetwork *nn)
{
	int i;

	if (nn->use_bias)
	{
		for (i = 0; i < nn->_n_neuro; i++)
		{
			nn->bias[i] = nn_gen_random() * 2;
		}
	}

	for (i = 0; i < nn->_n_weight; i++)
	{
		nn->weight[i] = nn_gen_random() * 2;
	}
}

void
nn_randomize_with_scale(NeuralNetwork *nn, float scale)
{
	int i;

	if (nn->use_bias)
	{
		for (i = 0; i < nn->_n_neuro; i++)
		{
			nn->bias[i] = nn_gen_random() * 2 * scale;
		}
	}

	for (i = 0; i < nn->_n_weight; i++)
	{
		nn->weight[i] = nn_gen_random() * 2 * scale ;
	}
}

void
nn_randomize_by_rate(NeuralNetwork *nn, float rate)
{
	int i;

	if (nn->use_bias)
	{
		for (i = 0; i < nn->_n_neuro; i++)
		{
			if (random_pick(rate))
				nn->bias[i] = nn_gen_random() * 2;
		}
	}

	for (i = 0; i < nn->_n_weight; i++)
	{
		if (random_pick(rate))
			nn->weight[i] = nn_gen_random() * 2;
	}
}

void
nn_randomize_with_scale_by_rate(NeuralNetwork *nn, float scale, float rate)
{
	int i;

	if (nn->use_bias)
	{
		for (i = 0; i < nn->_n_neuro; i++)
		{
			if (random_pick(rate))
				nn->bias[i] = nn_gen_random() * 2 * scale;
		}
	}

	for (i = 0; i < nn->_n_weight; i++)
	{
		if (random_pick(rate))
			nn->weight[i] = nn_gen_random() * 2 * scale;
	}

}

int
nn_save(NeuralNetwork *nn, const char * file_name)
{
	int ret = -1;
	FILE *f;

	f = fopen(file_name, "wb+");
	if (f == NULL)
		return -1;

	ret = nn_savef(nn, f);

	fclose(f);
	return ret;
}

NeuralNetwork *
nn_load(const char *file_name)
{
	NeuralNetwork *nn;
	FILE *f;

	f = fopen(file_name, "rb");
	if (f == NULL)
		return NULL;

	nn = nn_loadf(f);

	fclose(f);
	return nn;
}

int
nn_savef(NeuralNetwork *nn, FILE *f)
{
	/* write first informations */
	if (fwrite(&nn->n_input, sizeof(nn->n_input), 1, f) != 1)
		return -1;
	if (fwrite(&nn->n_output, sizeof(nn->n_output), 1, f) != 1)
		return -1;
	if (fwrite(&nn->n_hidden, sizeof(nn->n_hidden), 1, f) != 1)
		return -1;
	if (fwrite(&nn->n_neuro_per_hidden, sizeof(nn->n_neuro_per_hidden), 1, f) != 1)
		return -1;
	if (fwrite(&nn->use_bias, sizeof(nn->use_bias), 1, f) != 1)
		return -1;
	if (fwrite(&nn->act_func_type_hidden, sizeof(nn->act_func_type_hidden), 1, f) != 1)
		return -1;
	if (fwrite(&nn->act_func_type_output, sizeof(nn->act_func_type_output), 1, f) != 1)
		return -1;

	/* write weight and bias */
	if (fwrite(nn->weight, sizeof(float), nn->_n_weight, f) != nn->_n_weight)
		return -1;
	if (nn->use_bias)
	{
		if (fwrite(nn->bias, sizeof(float), nn->_n_neuro, f) != nn->_n_neuro)
			return -1;
	}

	return 0;
}

NeuralNetwork *
nn_loadf(FILE *f)
{
	NeuralNetwork *nn;

	nn = malloc(sizeof(*nn));

	/* read first informations */
	if (fread(&nn->n_input, sizeof(nn->n_input), 1, f) != 1)
		goto __error_1;
	if (fread(&nn->n_output, sizeof(nn->n_output), 1, f) != 1)
		goto __error_1;
	if (fread(&nn->n_hidden, sizeof(nn->n_hidden), 1, f) != 1)
		goto __error_1;
	if (fread(&nn->n_neuro_per_hidden, sizeof(nn->n_neuro_per_hidden), 1, f) != 1)
		goto __error_1;
	if (fread(&nn->use_bias, sizeof(nn->use_bias), 1, f) != 1)
		goto __error_1;
	if (fread(&nn->act_func_type_hidden, sizeof(nn->act_func_type_hidden), 1, f) != 1)
		goto __error_1;
	if (fread(&nn->act_func_type_output, sizeof(nn->act_func_type_output), 1, f) != 1)
		goto __error_1;

	nn->_n_neuro = nn->n_output + nn->n_hidden  * nn->n_neuro_per_hidden;
	nn->_n_weight = nn_compute_n_weight(nn);

	nn->weight = malloc(nn->_n_weight * sizeof(float));
	if (nn->use_bias)
		nn->bias = malloc(nn->_n_neuro * sizeof(float));
	nn->output = malloc(nn->_n_neuro * sizeof(float));
	nn->delta = malloc(nn->_n_neuro * sizeof(float));

	/* read weight and bias */
	if (fread(nn->weight, sizeof(float), nn->_n_weight, f) != nn->_n_weight)
		goto __error_2;
	if (nn->use_bias)
	{
		if (fread(nn->bias, sizeof(float), nn->_n_neuro, f) != nn->_n_neuro)
			goto __error_2;
	}

	return nn;

__error_1:
	free(nn->weight);
	free(nn->bias);
	free(nn->output);
	free(nn->delta);
__error_2:
	free(nn);
	return NULL;
}
