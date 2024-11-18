static void ptimer_reload(ptimer_state *s)

{

    if (s->delta == 0) {

        ptimer_trigger(s);

        s->delta = s->limit;

    }

    if (s->delta == 0 || s->period == 0) {

        fprintf(stderr, "Timer with period zero, disabling\n");

        s->enabled = 0;

        return;

    }



    s->last_event = s->next_event;

    s->next_event = s->last_event + s->delta * s->period;

    if (s->period_frac) {

        s->next_event += ((int64_t)s->period_frac * s->delta) >> 32;

    }

    timer_mod(s->timer, s->next_event);

}
