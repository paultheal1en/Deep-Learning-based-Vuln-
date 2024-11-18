static void openpic_gbl_write(void *opaque, hwaddr addr, uint64_t val,

                              unsigned len)

{

    OpenPICState *opp = opaque;

    IRQ_dst_t *dst;

    int idx;



    DPRINTF("%s: addr " TARGET_FMT_plx " <= %08x\n", __func__, addr, val);

    if (addr & 0xF)

        return;

    switch (addr) {

    case 0x00: /* Block Revision Register1 (BRR1) is Readonly */

        break;

    case 0x40:

    case 0x50:

    case 0x60:

    case 0x70:

    case 0x80:

    case 0x90:

    case 0xA0:

    case 0xB0:

        openpic_cpu_write_internal(opp, addr, val, get_current_cpu());

        break;

    case 0x1000: /* FREP */

        break;

    case 0x1020: /* GLBC */

        if (val & GLBC_RESET) {

            openpic_reset(&opp->busdev.qdev);

        }

        break;

    case 0x1080: /* VENI */

        break;

    case 0x1090: /* PINT */

        for (idx = 0; idx < opp->nb_cpus; idx++) {

            if ((val & (1 << idx)) && !(opp->pint & (1 << idx))) {

                DPRINTF("Raise OpenPIC RESET output for CPU %d\n", idx);

                dst = &opp->dst[idx];

                qemu_irq_raise(dst->irqs[OPENPIC_OUTPUT_RESET]);

            } else if (!(val & (1 << idx)) && (opp->pint & (1 << idx))) {

                DPRINTF("Lower OpenPIC RESET output for CPU %d\n", idx);

                dst = &opp->dst[idx];

                qemu_irq_lower(dst->irqs[OPENPIC_OUTPUT_RESET]);

            }

        }

        opp->pint = val;

        break;

    case 0x10A0: /* IPI_IPVP */

    case 0x10B0:

    case 0x10C0:

    case 0x10D0:

        {

            int idx;

            idx = (addr - 0x10A0) >> 4;

            write_IRQreg_ipvp(opp, opp->irq_ipi0 + idx, val);

        }

        break;

    case 0x10E0: /* SPVE */

        opp->spve = val & opp->vector_mask;

        break;

    default:

        break;

    }

}
