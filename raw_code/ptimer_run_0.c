void ptimer_run(ptimer_state *s, int oneshot)

{

    bool was_disabled = !s->enabled;



    if (was_disabled && s->period == 0) {

        fprintf(stderr, "Timer with period zero, disabling\n");

        return;

    }

    s->enabled = oneshot ? 2 : 1;

    if (was_disabled) {

        s->next_event = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

        ptimer_reload(s);

    }

}
