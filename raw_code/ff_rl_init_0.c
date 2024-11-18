av_cold void ff_rl_init(RLTable *rl,

                        uint8_t static_store[2][2 * MAX_RUN + MAX_LEVEL + 3])

{

    int8_t  max_level[MAX_RUN + 1], max_run[MAX_LEVEL + 1];

    uint8_t index_run[MAX_RUN + 1];

    int last, run, level, start, end, i;



    /* If table is static, we can quit if rl->max_level[0] is not NULL */

    if (static_store && rl->max_level[0])

        return;



    /* compute max_level[], max_run[] and index_run[] */

    for (last = 0; last < 2; last++) {

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

        for (i = start; i < end; i++) {

            run   = rl->table_run[i];

            level = rl->table_level[i];

            if (index_run[run] == rl->n)

                index_run[run] = i;

            if (level > max_level[run])

                max_level[run] = level;

            if (run > max_run[level])

                max_run[level] = run;

        }

        if (static_store)

            rl->max_level[last] = static_store[last];

        else

            rl->max_level[last] = av_malloc(MAX_RUN + 1);

        memcpy(rl->max_level[last], max_level, MAX_RUN + 1);

        if (static_store)

            rl->max_run[last]   = static_store[last] + MAX_RUN + 1;

        else

            rl->max_run[last]   = av_malloc(MAX_LEVEL + 1);

        memcpy(rl->max_run[last], max_run, MAX_LEVEL + 1);

        if (static_store)

            rl->index_run[last] = static_store[last] + MAX_RUN + MAX_LEVEL + 2;

        else

            rl->index_run[last] = av_malloc(MAX_RUN + 1);

        memcpy(rl->index_run[last], index_run, MAX_RUN + 1);

    }

}
