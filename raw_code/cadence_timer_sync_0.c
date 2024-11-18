static void cadence_timer_sync(CadenceTimerState *s)

{

    int i;

    int64_t r, x;

    int64_t interval = ((s->reg_count & COUNTER_CTRL_INT) ?

            (int64_t)s->reg_interval + 1 : 0x10000ULL) << 16;

    uint64_t old_time = s->cpu_time;



    s->cpu_time = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    DB_PRINT("cpu time: %lld ns\n", (long long)old_time);



    if (!s->cpu_time_valid || old_time == s->cpu_time) {

        s->cpu_time_valid = 1;

        return;

    }



    r = (int64_t)cadence_timer_get_steps(s, s->cpu_time - old_time);

    x = (int64_t)s->reg_value + ((s->reg_count & COUNTER_CTRL_DEC) ? -r : r);



    for (i = 0; i < 3; ++i) {

        int64_t m = (int64_t)s->reg_match[i] << 16;

        if (m > interval) {

            continue;

        }

        /* check to see if match event has occurred. check m +/- interval

         * to account for match events in wrap around cases */

        if (is_between(m, s->reg_value, x) ||

            is_between(m + interval, s->reg_value, x) ||

            is_between(m - interval, s->reg_value, x)) {

            s->reg_intr |= (2 << i);

        }

    }

    while (x < 0) {

        x += interval;

    }

    s->reg_value = (uint32_t)(x % interval);



    if (s->reg_value != x) {

        s->reg_intr |= (s->reg_count & COUNTER_CTRL_INT) ?

            COUNTER_INTR_IV : COUNTER_INTR_OV;

    }

    cadence_timer_update(s);

}
