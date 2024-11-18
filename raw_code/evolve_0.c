static void evolve(AVFilterContext *ctx)

{

    LifeContext *life = ctx->priv;

    int i, j;

    uint8_t *oldbuf = life->buf[ life->buf_idx];

    uint8_t *newbuf = life->buf[!life->buf_idx];



    enum { NW, N, NE, W, E, SW, S, SE };



    /* evolve the grid */

    for (i = 0; i < life->h; i++) {

        for (j = 0; j < life->w; j++) {

            int pos[8][2], n, alive, cell;

            if (life->stitch) {

                pos[NW][0] = (i-1) < 0 ? life->h-1 : i-1; pos[NW][1] = (j-1) < 0 ? life->w-1 : j-1;

                pos[N ][0] = (i-1) < 0 ? life->h-1 : i-1; pos[N ][1] =                         j  ;

                pos[NE][0] = (i-1) < 0 ? life->h-1 : i-1; pos[NE][1] = (j+1) == life->w ?  0 : j+1;

                pos[W ][0] =                         i  ; pos[W ][1] = (j-1) < 0 ? life->w-1 : j-1;

                pos[E ][0] =                         i  ; pos[E ][1] = (j+1) == life->w ? 0  : j+1;

                pos[SW][0] = (i+1) == life->h ?  0 : i+1; pos[SW][1] = (j-1) < 0 ? life->w-1 : j-1;

                pos[S ][0] = (i+1) == life->h ?  0 : i+1; pos[S ][1] =                         j  ;

                pos[SE][0] = (i+1) == life->h ?  0 : i+1; pos[SE][1] = (j+1) == life->w ?  0 : j+1;

            } else {

                pos[NW][0] = (i-1) < 0 ? -1        : i-1; pos[NW][1] = (j-1) < 0 ? -1        : j-1;

                pos[N ][0] = (i-1) < 0 ? -1        : i-1; pos[N ][1] =                         j  ;

                pos[NE][0] = (i-1) < 0 ? -1        : i-1; pos[NE][1] = (j+1) == life->w ? -1 : j+1;

                pos[W ][0] =                         i  ; pos[W ][1] = (j-1) < 0 ? -1        : j-1;

                pos[E ][0] =                         i  ; pos[E ][1] = (j+1) == life->w ? -1 : j+1;

                pos[SW][0] = (i+1) == life->h ? -1 : i+1; pos[SW][1] = (j-1) < 0 ? -1        : j-1;

                pos[S ][0] = (i+1) == life->h ? -1 : i+1; pos[S ][1] =                         j  ;

                pos[SE][0] = (i+1) == life->h ? -1 : i+1; pos[SE][1] = (j+1) == life->w ? -1 : j+1;

            }



            /* compute the number of live neighbor cells */

            n = (pos[NW][0] == -1 || pos[NW][1] == -1 ? 0 : oldbuf[pos[NW][0]*life->w + pos[NW][1]] == ALIVE_CELL) +

                (pos[N ][0] == -1 || pos[N ][1] == -1 ? 0 : oldbuf[pos[N ][0]*life->w + pos[N ][1]] == ALIVE_CELL) +

                (pos[NE][0] == -1 || pos[NE][1] == -1 ? 0 : oldbuf[pos[NE][0]*life->w + pos[NE][1]] == ALIVE_CELL) +

                (pos[W ][0] == -1 || pos[W ][1] == -1 ? 0 : oldbuf[pos[W ][0]*life->w + pos[W ][1]] == ALIVE_CELL) +

                (pos[E ][0] == -1 || pos[E ][1] == -1 ? 0 : oldbuf[pos[E ][0]*life->w + pos[E ][1]] == ALIVE_CELL) +

                (pos[SW][0] == -1 || pos[SW][1] == -1 ? 0 : oldbuf[pos[SW][0]*life->w + pos[SW][1]] == ALIVE_CELL) +

                (pos[S ][0] == -1 || pos[S ][1] == -1 ? 0 : oldbuf[pos[S ][0]*life->w + pos[S ][1]] == ALIVE_CELL) +

                (pos[SE][0] == -1 || pos[SE][1] == -1 ? 0 : oldbuf[pos[SE][0]*life->w + pos[SE][1]] == ALIVE_CELL);

            cell  = oldbuf[i*life->w + j];

            alive = 1<<n & (cell == ALIVE_CELL ? life->stay_rule : life->born_rule);

            if (alive)     *newbuf = ALIVE_CELL; // new cell is alive

            else if (cell) *newbuf = cell - 1;   // new cell is dead and in the process of mold

            else           *newbuf = 0;          // new cell is definitely dead

            av_dlog(ctx, "i:%d j:%d live_neighbors:%d cell:%d -> cell:%d\n", i, j, n, cell, *newbuf);

            newbuf++;

        }

    }



    life->buf_idx = !life->buf_idx;

}
