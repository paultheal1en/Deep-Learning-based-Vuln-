static void arm_timer_recalibrate(arm_timer_state *s, int reload)

{

    uint32_t limit;



    if ((s->control & TIMER_CTRL_PERIODIC) == 0) {

        /* Free running.  */

        if (s->control & TIMER_CTRL_32BIT)

            limit = 0xffffffff;

        else

            limit = 0xffff;

    } else {

          /* Periodic.  */

          limit = s->limit;

    }

    ptimer_set_limit(s->timer, limit, reload);

}
