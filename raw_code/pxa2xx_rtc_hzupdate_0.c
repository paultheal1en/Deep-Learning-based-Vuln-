static void pxa2xx_rtc_hzupdate(PXA2xxRTCState *s)

{

    int64_t rt = qemu_get_clock(rt_clock);

    s->last_rcnr += ((rt - s->last_hz) << 15) /

            (1000 * ((s->rttr & 0xffff) + 1));

    s->last_rdcr += ((rt - s->last_hz) << 15) /

            (1000 * ((s->rttr & 0xffff) + 1));

    s->last_hz = rt;

}
