#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>
#include <ctype.h>
#include <time.h>
#include <pthread.h>
#include <string.h>

#include "snake_game.h"
#include "neural_network.h"
#include "neural_network_elite.h"

#define AI_STATUS_FILE	"snake.status"
#define MUTATION_RATE	0.1f
#define ELITE_THRESHOLD	0.8

#define GAME_X			32
#define GAME_Y			16
#define GAME_MAX_STEP	500

#define GAME_RANDOM_MAP	0
#define GAME_SEED		1128

typedef struct AIStatus{
	int gen;
	float best_performance;
	float best_score;
	NNEliteList elite_list;
} AIStatus;

typedef struct Param {
	int game_seed;
	int game_rand_map;
	float mutation_rate;
	int progress;
	int replay;
	const char *status_f;
} Param;

static AIStatus status;
static Param param = {
	/* Some default values */
	.game_seed = GAME_SEED,
	.game_rand_map = GAME_RANDOM_MAP,
	.mutation_rate = MUTATION_RATE,
	.progress = 0,
	.replay = 0,
	.status_f = AI_STATUS_FILE
};

static int should_stop = 0;

static pthread_t display_thread;
/* Prevent the neural network get destroyed during duplication before showcase */
static pthread_mutex_t status_lock = PTHREAD_MUTEX_INITIALIZER;

static void signal_handler(int sig);

static int parse_opt(int argc, char **argv);

static int ai_status_init(const char *file_name, AIStatus *status);
static int ai_status_exit(const char *file_name, AIStatus *status);

static void *_display_thread_func(void *arg);
static int _find_max_in_array(float *arr, int len);
static void _ai_run_n_games(NeuralNetwork *nn, int n, int demo, float *avg_performance, float *avg_score);

static void ai_progress(void);
static void ai_replay(void);

static int
parse_opt(int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, "hrs:f:m:PR")) != -1)
	{
		switch (c)
		{
			case 'R':
				param.replay = 1;
				break;
			case 'P':
				param.progress = 1;
				break;
			case 'r':
				param.game_rand_map = 1;
				break;
			case 's':
				param.game_seed = atoi(optarg);
				break;
			case 'f':
				param.status_f = optarg;
				break;
			case 'm':
				param.mutation_rate = atof(optarg);
				break;
			case 'h':
			default:
				/* Print help */
				printf("%s\n"
						"    -R replay the best neural network result.\n"
						"    -P progress the training.\n"
						"    -s <game_seed> for non-random map\n"
						"    -r for randomized map generation\n"
						"    -f <file_name> to save file\n",
						argv[0]);
				exit(0);
		}
	}

	return 0;
}

static void
signal_handler(int sig)
{
	should_stop = 1;
}

static int
ai_status_init(const char *file_name, AIStatus *status)
{
	FILE *f;
	int ret;

	ret = -1;
	f = fopen(file_name, "rb");
	if (f == NULL)
		return -1;

	if (fread(&status->gen, sizeof(status->gen), 1, f) != 1)
		goto __exit;

	if (fread(&status->best_performance, sizeof(status->best_performance), 1, f) != 1)
		goto __exit;

	if (fread(&status->best_score, sizeof(status->best_score), 1, f) != 1)
		goto __exit;

	if (nn_elites_loadf(&status->elite_list, f))
		goto __exit;

	ret = 0;
__exit:
	fclose(f);
	return ret;
}

static int
ai_status_exit(const char *file_name, AIStatus *status)
{
	FILE *f;
	int ret;

	ret = -1;
	f = fopen(file_name, "wb");
	if (f == NULL)
		return -1;

	if (fwrite(&status->gen, sizeof(status->gen), 1, f) != 1)
		goto __exit;

	if (fwrite(&status->best_performance, sizeof(status->best_performance), 1, f) != 1)
		goto __exit;

	if (fwrite(&status->best_score, sizeof(status->best_score), 1, f) != 1)
		goto __exit;

	if (nn_elites_savef(&status->elite_list, f))
		goto __exit;

	ret = 0;
__exit:
	fclose(f);
	return ret;
}

static int
_find_max_in_array(float *arr, int len)
{
	int i;
	float f_max = arr[0];
	int i_max = 0;

	for (i = 1; i < len; i++)
	{
		if (f_max < arr[i])
		{
			f_max = arr[i];
			i_max = i;
		}
	}

	return i_max;
}

static void *
_display_thread_func(void *arg)
{
	NeuralNetwork *the_best;
	NeuralNetwork *nn;
	int n_elite;
	float performance;
	float score;

	while (!should_stop)
	{
		pthread_mutex_lock(&status_lock);
		the_best = nn_elites_get_best(&status.elite_list);
		nn = nn_duplicate(the_best);
		pthread_mutex_unlock(&status_lock);
		if (nn == NULL)
		{
			printf("Waiting for the best to be generated.\n");
			usleep(1000000);
			continue;
		}

		_ai_run_n_games(nn,
				1,
				1,
				&performance,
				&score);

		nn_free(nn);
	}

	pthread_exit(NULL);
}

static void
_ai_run_n_games(NeuralNetwork *nn, int n, int demo, float *avg_performance, float *avg_score)
{
	int i;
	SnakeGame *game = NULL;
	float input[8];
	float *output;
	int dir;

	*avg_score = 0;
	*avg_performance = 0;
	for (i = 0; i < n; i++)
	{
		game = snake_game_create(GAME_X,
				GAME_Y,
				8,
				GAME_MAX_STEP,
				param.game_rand_map ? rand() : param.game_seed);
		if (demo)
			snake_game_show(game);
		while (!snake_game_is_over(game) && !should_stop)
		{
			input[0] = game->dist_to_hit[0];
			input[1] = game->dist_to_hit[1];
			input[2] = game->dist_to_hit[2];
			input[3] = game->dist_to_hit[3];
			input[4] = game->dist_to_food[0];
			input[5] = game->dist_to_food[1];
			input[6] = game->dist_to_food[2];
			input[7] = game->dist_to_food[3];

			output = nn_run(nn, input);
			dir = _find_max_in_array(output, 4);

			snake_game_set_direction(game, dir, 1);
			snake_game_update(game, 1, demo ? 1 : 0);

			if (demo)
			{
				pthread_mutex_lock(&status_lock);
				printf("Save file: \"%s\"\n", param.status_f);
				printf("Mutation rate: %f\n", param.mutation_rate);
				if (param.game_rand_map)
					printf("Game seed: Randomized\n");
				else
					printf("Game seed: %d\n", param.game_seed);

				printf("Current generation: %d\n", status.gen);
				printf("Best performance: %.2f\n", status.best_performance);
				nn_elite_show(&status.elite_list);
				pthread_mutex_unlock(&status_lock);
				usleep(1000000 / 16);
			}
		}

		*avg_score += snake_game_get_score(game);
		*avg_performance += snake_game_get_performance(game);
		snake_game_free(game);
	}

	*avg_score /= (float)n;
	*avg_performance /= (float)n;
}

static void
ai_progress(void)
{
	NeuralNetwork *best = NULL;
	NeuralNetwork *nn = NULL;

	float performance;
	float score;

	pthread_create(&display_thread, NULL, _display_thread_func, NULL);

	while (!should_stop)
	{
		pthread_mutex_lock(&status_lock);
		best = nn_elites_get_best(&status.elite_list);
		if (best == NULL)
		{
			nn = nn_create(8,
					4,
					2,
					8,
					0,
					ACT_FUNC_TYPE_LINEAR,
					ACT_FUNC_TYPE_LINEAR);
		}
		else
		{
			/* produce from elites */
			NeuralNetwork *parent_a;
			NeuralNetwork *parent_b;

			/* 1. Choose parents */
			//parent_a = nn_elites_pick_by_random(&elite_list, NULL);
			parent_a = best;
			parent_b = nn_elites_pick_by_random(&status.elite_list, parent_a);

			/* 2. Produce child */
			nn = nn_produce(parent_a, parent_b);

			/* 3. Mutate */
			nn_randomize_by_rate(nn, param.mutation_rate);
		}
		pthread_mutex_unlock(&status_lock);

		_ai_run_n_games(nn,
				param.game_rand_map ? 10 : 1,
				0,
				&performance,
				&score);

		/* Count generation */
		if (performance > status.best_performance)
		{
			best = nn;
			pthread_mutex_lock(&status_lock);
			status.gen++;
			status.best_performance = performance;
			status.best_score = score;
			pthread_mutex_unlock(&status_lock);
		}

		if (performance > status.best_performance * ELITE_THRESHOLD)
		{
			pthread_mutex_lock(&status_lock);
			nn_elites_add(&status.elite_list, nn, performance);
			pthread_mutex_unlock(&status_lock);
		}
		else
		{
			nn_free(nn);
		}
	}

	pthread_join(display_thread, NULL);
}

static void
ai_replay(void)
{
	NeuralNetwork *the_best = NULL;
	NeuralNetwork *nn = NULL;
	float performance;
	float score = 0;

	the_best = nn_elites_get_best(&status.elite_list);
	nn = nn_duplicate(the_best);
	if (nn == NULL)
	{
		printf("Failed to load.\n");
		return;
	}

	_ai_run_n_games(nn,
			1,
			1,
			&performance,
			&score);

	nn_free(nn);
}

int main(int argc, char **argv)
{
	/* Init */
	srand(time(NULL));

	signal(SIGINT, signal_handler);

	if (argc < 2)
		return 0;

	parse_opt(argc, argv);

	if (ai_status_init(param.status_f, &status))
	{
		/* Initialize everything */
		status.gen = 0;
		status.best_performance = 0;
		status.best_score = 0;
		nn_elites_init_list(&status.elite_list, 10);
	}

	if (param.progress)
	{
		ai_progress();
		ai_status_exit(param.status_f, &status);
	}
	else if (param.replay)
	{
		ai_replay();
	}

	nn_elites_clear(&status.elite_list);

	return 0;
}
