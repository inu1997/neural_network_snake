#include "neural_network_elite.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct _NNEliteList{
	NeuralNetwork *nn;
	float goodness;
	void *prev;
	void *next;
} _NNEliteList;

static void _nn_elite_free(_NNEliteList *el);

static void _nn_elist_list_link(_NNEliteList *el1, _NNEliteList *el2);

static void
_nn_elite_free(_NNEliteList *el)
{
	nn_free(el->nn);
	free(el);
}

static void
_nn_elist_list_link(_NNEliteList *el1, _NNEliteList *el2)
{
	el1->next = el2;
	el2->prev = el1;
}

void
nn_elites_init_list(NNEliteList *list, int max_len)
{
	list->list_head = NULL;
	list->max_len = max_len;
}

void
nn_elites_add(NNEliteList *list, NeuralNetwork *nn, float goodness)
{
	_NNEliteList *new_node;
	_NNEliteList *ptr;

	new_node = malloc(sizeof(*new_node));
	new_node->goodness = goodness;
	new_node->nn = nn;
	new_node->next = NULL;
	new_node->prev = NULL;

	if (list->list_head == NULL)
	{
		/* First add */
		new_node->next = new_node;
		new_node->prev = new_node;
		list->list_head = new_node;
		return;
	}

	/* Find where to insert the node so the best could be in the front of the list */
	ptr = list->list_head;
	do
	{
		if (ptr->goodness < goodness)
			break;
		ptr = ptr->next;
	}
	while (ptr != list->list_head);

	/* new_node is better than ptr node */
	_nn_elist_list_link(ptr->prev, new_node);
	_nn_elist_list_link(new_node, ptr);

	/* Only 2 possible situation ptr is list_head,
	 * 1. new node is better. so it breaks and put behind the list_head.
	 * 2. new node is the worst. so it's put to the last position(behind the list_head)
	 */
	if (ptr == list->list_head &&
		ptr->goodness < goodness)
		list->list_head = new_node;

	if (list->max_len < nn_elites_get_count(list))
	{
		_NNEliteList *worst;

		worst = ((_NNEliteList*)list->list_head)->prev;

		/* The worst gets freed */
		((_NNEliteList *)worst->prev)->next = worst->next;
		((_NNEliteList *)worst->next)->prev = worst->prev;
		_nn_elite_free(worst);
	}
}

void
nn_elites_clear(NNEliteList *list)
{
	_NNEliteList *el;

	if (nn_elites_get_count(list) == 0)
		return;

	el = list->list_head;
	while (el->next != list->list_head)
	{
		el = el->next;
		_nn_elite_free(el->prev);
	};

	_nn_elite_free(el);

	list->list_head = NULL;
}

NeuralNetwork *
nn_elites_pick_by_random(NNEliteList *list, NeuralNetwork *dont_pick)
{
	int i;
	_NNEliteList *el;
	int e_cnt;

	e_cnt = nn_elites_get_count(list);
	if (e_cnt == 0)
		return NULL;

	el = list->list_head;
	do
	{
		i = rand() % e_cnt;
		el = list->list_head;
		while (i)
		{
			el = el->next;
			i--;
		}
		if (e_cnt == 1)
			break;
	}
	while (el->nn == dont_pick);

	return el->nn;
}

NeuralNetwork *
nn_elites_get_best(NNEliteList *list)
{
	if (list->list_head == NULL)
		return NULL;

	return ((_NNEliteList*)list->list_head)->nn;
}

int
nn_elites_get_count(NNEliteList *list)
{
	int cnt;
	_NNEliteList *el;

	el = list->list_head;
	if (el == NULL)
		return 0;

	cnt = 0;
	do
	{
		cnt++;
		el = el->next;
	} while (el != list->list_head);

	return cnt;
}

int
nn_elites_save(NNEliteList *list, const char *file_name)
{
	int ret;
	FILE *f;
	f = fopen(file_name, "wb");
	if (f == NULL)
		return -1;

	ret = nn_elites_savef(list, f);

	fclose(f);
	return ret;
}

int
nn_elites_load(NNEliteList *list, const char *file_name)
{
	int ret;
	FILE *f;
	f = fopen(file_name, "rb");
	if (f == NULL)
		return -1;

	ret = nn_elites_loadf(list, f);

	fclose(f);
	return ret;
}

int
nn_elites_savef(NNEliteList *list, FILE *f)
{
	int cnt;
	_NNEliteList *el;

	if (fwrite(&list->max_len, sizeof(list->max_len), 1, f) != 1)
		return -1;

	cnt = nn_elites_get_count(list);
	if (fwrite(&cnt, sizeof(cnt), 1, f) != 1)
		return -1;

	el = list->list_head;
	do
	{
		if (nn_savef(el->nn, f))
			return -1;

		if (fwrite(&el->goodness, sizeof(el->goodness), 1, f) != 1)
			return -1;

		el = el->next;
	} while (el != list->list_head);

	return 0;
}

int
nn_elites_loadf(NNEliteList *list, FILE *f)
{
	int cnt;
	int i;
	float goodness;
	NeuralNetwork *nn;

	if (fread(&list->max_len, sizeof(list->max_len), 1, f) != 1)
		return -1;

	cnt = nn_elites_get_count(list);
	if (fread(&cnt, sizeof(cnt), 1, f) != 1)
		return -1;

	for (i = 0; i < cnt; i++)
	{
		nn = nn_loadf(f);
		if (nn == NULL)
			return -1;

		if (fread(&goodness, sizeof(goodness), 1, f) != 1)
			return -1;

		nn_elites_add(list, nn, goodness);
	}

	return 0;
}

void
nn_elite_show(NNEliteList *list)
{
	int i;
	_NNEliteList *el;

	el = list->list_head;
	if (el == NULL)
		return;

	i = 0;
	do
	{
		i++;
		printf("Elite %d goodness: %6.2f\n", i, el->goodness);
		el = el->next;
	} while (el != list->list_head);
}

