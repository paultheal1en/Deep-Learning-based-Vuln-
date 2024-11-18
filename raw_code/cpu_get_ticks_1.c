int64_t cpu_get_ticks(void)

{

    if (use_icount) {

        return cpu_get_icount();

    }

    if (!timers_state.cpu_ticks_enabled) {

        return timers_state.cpu_ticks_offset;

    } else {

        int64_t ticks;

        ticks = cpu_get_real_ticks();

        if (timers_state.cpu_ticks_prev > ticks) {

            /* Note: non increasing ticks may happen if the host uses

               software suspend */

            timers_state.cpu_ticks_offset += timers_state.cpu_ticks_prev - ticks;

        }

        timers_state.cpu_ticks_prev = ticks;

        return ticks + timers_state.cpu_ticks_offset;

    }

}
