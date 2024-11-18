static uint32_t pxa2xx_rtc_read(void *opaque, target_phys_addr_t addr)

{

    PXA2xxRTCState *s = (PXA2xxRTCState *) opaque;



    switch (addr) {

    case RTTR:

        return s->rttr;

    case RTSR:

        return s->rtsr;

    case RTAR:

        return s->rtar;

    case RDAR1:

        return s->rdar1;

    case RDAR2:

        return s->rdar2;

    case RYAR1:

        return s->ryar1;

    case RYAR2:

        return s->ryar2;

    case SWAR1:

        return s->swar1;

    case SWAR2:

        return s->swar2;

    case PIAR:

        return s->piar;

    case RCNR:

        return s->last_rcnr + ((qemu_get_clock(rt_clock) - s->last_hz) << 15) /

                (1000 * ((s->rttr & 0xffff) + 1));

    case RDCR:

        return s->last_rdcr + ((qemu_get_clock(rt_clock) - s->last_hz) << 15) /

                (1000 * ((s->rttr & 0xffff) + 1));

    case RYCR:

        return s->last_rycr;

    case SWCR:

        if (s->rtsr & (1 << 12))

            return s->last_swcr + (qemu_get_clock(rt_clock) - s->last_sw) / 10;

        else

            return s->last_swcr;

    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

    return 0;

}
