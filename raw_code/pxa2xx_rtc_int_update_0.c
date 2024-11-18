static inline void pxa2xx_rtc_int_update(PXA2xxState *s)

{

    qemu_set_irq(s->pic[PXA2XX_PIC_RTCALARM], !!(s->rtsr & 0x2553));

}
