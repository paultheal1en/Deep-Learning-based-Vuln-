static void imx_ccm_write(void *opaque, target_phys_addr_t offset,

                          uint64_t value, unsigned size)

{

    IMXCCMState *s = (IMXCCMState *)opaque;



    DPRINTF("write(offset=%x, value = %x)\n",

            offset >> 2, (unsigned int)value);

    switch (offset >> 2) {

    case 0:

        s->ccmr = CCMR_FPMF | (value & 0x3b6fdfff);

        break;

    case 1:

        s->pdr0 = value & 0xff9f3fff;

        break;

    case 2:

        s->pdr1 = value;

        break;

    case 4:

        s->mpctl = value & 0xbfff3fff;

        break;

    case 6:

        s->spctl = value & 0xbfff3fff;

        break;

    case 8:

        s->cgr[0] = value;

        return;

    case 9:

        s->cgr[1] = value;

        return;

    case 10:

        s->cgr[2] = value;

        return;



    default:

        return;

    }

    update_clocks(s);

}
