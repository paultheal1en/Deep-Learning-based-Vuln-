static inline void pit_load_count(PITChannelState *s, int val)

{

    if (val == 0)

        val = 0x10000;

    s->count_load_time = cpu_get_ticks();

    s->count_last_edge_check_time = s->count_load_time;

    s->count = val;

    if (s == &pit_channels[0] && val <= pit_min_timer_count) {

        fprintf(stderr, 

                "\nWARNING: qemu: on your system, accurate timer emulation is impossible if its frequency is more than %d Hz. If using a 2.5.xx Linux kernel, you must patch asm/param.h to change HZ from 1000 to 100.\n\n", 

                PIT_FREQ / pit_min_timer_count);

    }

}
