static void mv88w8618_pit_write(void *opaque, target_phys_addr_t offset,

                                uint32_t value)

{

    mv88w8618_pit_state *s = opaque;

    mv88w8618_timer_state *t;

    int i;



    switch (offset) {

    case MP_PIT_TIMER1_LENGTH ... MP_PIT_TIMER4_LENGTH:

        t = &s->timer[offset >> 2];

        t->limit = value;

        ptimer_set_limit(t->ptimer, t->limit, 1);

        break;



    case MP_PIT_CONTROL:

        for (i = 0; i < 4; i++) {

            if (value & 0xf) {

                t = &s->timer[i];

                ptimer_set_limit(t->ptimer, t->limit, 0);

                ptimer_set_freq(t->ptimer, t->freq);

                ptimer_run(t->ptimer, 0);

            }

            value >>= 4;

        }

        break;



    case MP_BOARD_RESET:

        if (value == MP_BOARD_RESET_MAGIC) {

            qemu_system_reset_request();

        }

        break;

    }

}
