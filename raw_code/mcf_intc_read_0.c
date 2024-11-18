static uint64_t mcf_intc_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    int offset;

    mcf_intc_state *s = (mcf_intc_state *)opaque;

    offset = addr & 0xff;

    if (offset >= 0x40 && offset < 0x80) {

        return s->icr[offset - 0x40];

    }

    switch (offset) {

    case 0x00:

        return (uint32_t)(s->ipr >> 32);

    case 0x04:

        return (uint32_t)s->ipr;

    case 0x08:

        return (uint32_t)(s->imr >> 32);

    case 0x0c:

        return (uint32_t)s->imr;

    case 0x10:

        return (uint32_t)(s->ifr >> 32);

    case 0x14:

        return (uint32_t)s->ifr;

    case 0xe0: /* SWIACK.  */

        return s->active_vector;

    case 0xe1: case 0xe2: case 0xe3: case 0xe4:

    case 0xe5: case 0xe6: case 0xe7:

        /* LnIACK */

        hw_error("mcf_intc_read: LnIACK not implemented\n");

    default:

        return 0;

    }

}
