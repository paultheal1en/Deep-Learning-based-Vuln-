static void omap_lpg_update(struct omap_lpg_s *s)

{

    int64_t on, period = 1, ticks = 1000;

    static const int per[8] = { 1, 2, 4, 8, 12, 16, 20, 24 };



    if (~s->control & (1 << 6))					/* LPGRES */

        on = 0;

    else if (s->control & (1 << 7))				/* PERM_ON */

        on = period;

    else {

        period = muldiv64(ticks, per[s->control & 7],		/* PERCTRL */

                        256 / 32);

        on = (s->clk && s->power) ? muldiv64(ticks,

                        per[(s->control >> 3) & 7], 256) : 0;	/* ONCTRL */

    }



    timer_del(s->tm);

    if (on == period && s->on < s->period)

        printf("%s: LED is on\n", __FUNCTION__);

    else if (on == 0 && s->on)

        printf("%s: LED is off\n", __FUNCTION__);

    else if (on && (on != s->on || period != s->period)) {

        s->cycle = 0;

        s->on = on;

        s->period = period;

        omap_lpg_tick(s);

        return;

    }



    s->on = on;

    s->period = period;

}
