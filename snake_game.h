#ifndef __SNAKE_GAME_H
#define __SNAKE_GAME_H

#include <sys/time.h>
#include "mtwister.h"

/***************************** Game Configuration *****************************/

#ifndef TV_USEC_DIFF
#define TV_USEC_DIFF(old_tv, new_tv)	( \
		(long)((long)(new_tv)->tv_sec - (long)(old_tv)->tv_sec) * 1000000 + \
		(long)((long)(new_tv)->tv_usec - (long)(old_tv)->tv_usec) \
		)
#endif

typedef struct {
	int x;
	int y;
} Point;

typedef enum {
	DIRECTION_UP,
	DIRECTION_DOWN,
	DIRECTION_LEFT,
	DIRECTION_RIGHT,
	DIRECTION_NONE,
} DIRECTION;

typedef struct {
	/* Game info */
	struct timeval tv_last_step;
	int step_per_sec;
	int init_step_to_food;
	int total_step_to_food;
	int total_step_used;
	int max_step;
	int game_over;
	char game_over_reason[64];
	int size_x;
	int size_y;

	/* Snake */
	Point *snake_body;
	int snake_len;			/* Length of snake */
	DIRECTION snake_dir;	/* Direction for snake to move */
	int snake_step_remain;	/* How many steps remain before snake die because it didn't eat */

	/* Point */
	Point pt;

	/* Display */
	char *display_bg;	/* Background buffer before showing in terminal */
	char *display_fg;	/* Foreground buffer after showing in terminal */
	MTRand mtrand;

	/* Feed to neural network */
	int dist_to_hit[4];
	int dist_to_food[4];
} SnakeGame;

SnakeGame *snake_game_create(int x, int y, int step_per_sec, int max_step, int seed);

void snake_game_free(SnakeGame *game);

void snake_game_set_direction(SnakeGame *game, DIRECTION dir, int prevent_suicide);

void snake_game_update(SnakeGame *game, int no_wait, int update_display);

void snake_game_show(SnakeGame *game);

void snake_game_over(SnakeGame *game, const char *reason);

int snake_game_is_over(SnakeGame *game);

int snake_game_get_score(SnakeGame *game);

const char *snake_game_get_game_over_reason(SnakeGame *game);

float snake_game_get_performance(SnakeGame *game);

#endif /* __SNAKE_GAME_H */
