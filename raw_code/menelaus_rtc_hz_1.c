static void menelaus_rtc_hz(void *opaque)

{

    struct menelaus_s *s = (struct menelaus_s *) opaque;



    s->rtc.next_comp --;

    s->rtc.alm_sec --;

    s->rtc.next += 1000;

    qemu_mod_timer(s->rtc.hz, s->rtc.next);

    if ((s->rtc.ctrl >> 3) & 3) {				/* EVERY */

        menelaus_rtc_update(s);

        if (((s->rtc.ctrl >> 3) & 3) == 1 && !s->rtc.tm.tm_sec)

            s->status |= 1 << 8;				/* RTCTMR */

        else if (((s->rtc.ctrl >> 3) & 3) == 2 && !s->rtc.tm.tm_min)

            s->status |= 1 << 8;				/* RTCTMR */

        else if (!s->rtc.tm.tm_hour)

            s->status |= 1 << 8;				/* RTCTMR */

    } else

        s->status |= 1 << 8;					/* RTCTMR */

    if ((s->rtc.ctrl >> 1) & 1) {				/* RTC_AL_EN */

        if (s->rtc.alm_sec == 0)

            s->status |= 1 << 9;				/* RTCALM */

        /* TODO: wake-up */

    }

    if (s->rtc.next_comp <= 0) {

        s->rtc.next -= muldiv64((int16_t) s->rtc.comp, 1000, 0x8000);

        s->rtc.next_comp = 3600;

    }

    menelaus_update(s);

}
