static void omap_mcbsp_source_tick(void *opaque)

{

    struct omap_mcbsp_s *s = (struct omap_mcbsp_s *) opaque;

    static const int bps[8] = { 0, 1, 1, 2, 2, 2, -255, -255 };



    if (!s->rx_rate)

        return;

    if (s->rx_req)

        printf("%s: Rx FIFO overrun\n", __FUNCTION__);



    s->rx_req = s->rx_rate << bps[(s->rcr[0] >> 5) & 7];



    omap_mcbsp_rx_newdata(s);

    timer_mod(s->source_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +

                   NANOSECONDS_PER_SECOND);

}
