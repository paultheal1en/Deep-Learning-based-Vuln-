static void omap_lpg_tick(void *opaque)

{

    struct omap_lpg_s *s = opaque;



    if (s->cycle)

        timer_mod(s->tm, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + s->period - s->on);

    else

        timer_mod(s->tm, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + s->on);



    s->cycle = !s->cycle;

    printf("%s: LED is %s\n", __FUNCTION__, s->cycle ? "on" : "off");

}
