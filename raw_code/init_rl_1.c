void init_rl(RLTable *rl)

{

    int8_t max_level[MAX_RUN+1], max_run[MAX_LEVEL+1];

    uint8_t index_run[MAX_RUN+1];

    int last, run, level, start, end, i;



    /* compute max_level[], max_run[] and index_run[] */

    for(last=0;last<2;last++) {

        if (last == 0) {

            start = 0;

            end = rl->last;

        } else {

            start = rl->last;

            end = rl->n;

        }



        memset(max_level, 0, MAX_RUN + 1);

        memset(max_run, 0, MAX_LEVEL + 1);

        memset(index_run, rl->n, MAX_RUN + 1);

        for(i=start;i<end;i++) {

            run = rl->table_run[i];

            level = rl->table_level[i];

            if (index_run[run] == rl->n)

                index_run[run] = i;

            if (level > max_level[run])

                max_level[run] = level;

            if (run > max_run[level])

                max_run[level] = run;

        }

        rl->max_level[last] = av_malloc(MAX_RUN + 1);

        memcpy(rl->max_level[last], max_level, MAX_RUN + 1);

        rl->max_run[last] = av_malloc(MAX_LEVEL + 1);

        memcpy(rl->max_run[last], max_run, MAX_LEVEL + 1);

        rl->index_run[last] = av_malloc(MAX_RUN + 1);

        memcpy(rl->index_run[last], index_run, MAX_RUN + 1);

    }

}
