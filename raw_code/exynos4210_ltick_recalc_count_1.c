static void exynos4210_ltick_recalc_count(struct tick_timer *s)

{

    uint64_t to_count;



    if ((s->cnt_run && s->last_tcnto) || (s->int_run && s->last_icnto)) {

        /*

         * one or both timers run and not counted to the end;

         * distance is not passed, recalculate with last_tcnto * last_icnto

         */



        if (s->last_tcnto) {

            to_count = s->last_tcnto * s->last_icnto;

        } else {

            to_count = s->last_icnto;

        }

    } else {

        /* distance is passed, recalculate with tcnto * icnto */

        if (s->icntb) {

            s->distance = s->tcntb * s->icntb;

        } else {

            s->distance = s->tcntb;

        }



        to_count = s->distance;

        s->progress = 0;

    }



    if (to_count > MCT_LT_COUNTER_STEP) {

        /* count by step */

        s->count = MCT_LT_COUNTER_STEP;

    } else {

        s->count = to_count;

    }

}
