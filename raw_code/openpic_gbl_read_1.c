static uint64_t openpic_gbl_read(void *opaque, hwaddr addr, unsigned len)

{

    OpenPICState *opp = opaque;

    uint32_t retval;



    DPRINTF("%s: addr " TARGET_FMT_plx "\n", __func__, addr);

    retval = 0xFFFFFFFF;

    if (addr & 0xF)

        return retval;

    switch (addr) {

    case 0x1000: /* FREP */

        retval = opp->frep;

        break;

    case 0x1020: /* GLBC */

        retval = opp->glbc;

        break;

    case 0x1080: /* VENI */

        retval = opp->veni;

        break;

    case 0x1090: /* PINT */

        retval = 0x00000000;

        break;

    case 0x00: /* Block Revision Register1 (BRR1) */

        retval = opp->brr1;

        break;

    case 0x40:

    case 0x50:

    case 0x60:

    case 0x70:

    case 0x80:

    case 0x90:

    case 0xA0:

    case 0xB0:

        retval = openpic_cpu_read_internal(opp, addr, get_current_cpu());

        break;

    case 0x10A0: /* IPI_IPVP */

    case 0x10B0:

    case 0x10C0:

    case 0x10D0:

        {

            int idx;

            idx = (addr - 0x10A0) >> 4;

            retval = read_IRQreg_ipvp(opp, opp->irq_ipi0 + idx);

        }

        break;

    case 0x10E0: /* SPVE */

        retval = opp->spve;

        break;

    default:

        break;

    }

    DPRINTF("%s: => %08x\n", __func__, retval);



    return retval;

}
