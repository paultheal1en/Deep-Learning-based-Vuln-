static void imx_avic_write(void *opaque, target_phys_addr_t offset,

                          uint64_t val, unsigned size)

{

    IMXAVICState *s = (IMXAVICState *)opaque;



    /* Vector Registers not yet supported */

    if (offset >= 0x100 && offset <= 0x2fc) {

        IPRINTF("imx_avic_write to vector register %d ignored\n",

                (unsigned int)((offset - 0x100) >> 2));

        return;

    }



    DPRINTF("imx_avic_write(0x%x) = %x\n",

            (unsigned int)offset>>2, (unsigned int)val);

    switch (offset >> 2) {

    case 0: /* Interrupt Control Register, INTCNTL */

        s->intcntl = val & (ABFEN | NIDIS | FIDIS | NIAD | FIAD | NM);

        if (s->intcntl & ABFEN) {

            s->intcntl &= ~(val & ABFLAG);

        }

        break;



    case 1: /* Normal Interrupt Mask Register, NIMASK */

        s->intmask = val & 0x1f;

        break;



    case 2: /* Interrupt Enable Number Register, INTENNUM */

        DPRINTF("enable(%d)\n", (int)val);

        val &= 0x3f;

        s->enabled |= (1ULL << val);

        break;



    case 3: /* Interrupt Disable Number Register, INTDISNUM */

        DPRINTF("disable(%d)\n", (int)val);

        val &= 0x3f;

        s->enabled &= ~(1ULL << val);

        break;



    case 4: /* Interrupt Enable Number Register High */

        s->enabled = (s->enabled & 0xffffffffULL) | (val << 32);

        break;



    case 5: /* Interrupt Enable Number Register Low */

        s->enabled = (s->enabled & 0xffffffff00000000ULL) | val;

        break;



    case 6: /* Interrupt Type Register High */

        s->is_fiq = (s->is_fiq & 0xffffffffULL) | (val << 32);

        break;



    case 7: /* Interrupt Type Register Low */

        s->is_fiq = (s->is_fiq & 0xffffffff00000000ULL) | val;

        break;



    case 8: /* Normal Interrupt Priority Register 7 */

    case 9: /* Normal Interrupt Priority Register 6 */

    case 10:/* Normal Interrupt Priority Register 5 */

    case 11:/* Normal Interrupt Priority Register 4 */

    case 12:/* Normal Interrupt Priority Register 3 */

    case 13:/* Normal Interrupt Priority Register 2 */

    case 14:/* Normal Interrupt Priority Register 1 */

    case 15:/* Normal Interrupt Priority Register 0 */

        s->prio[15-(offset>>2)] = val;

        break;



        /* Read-only registers, writes ignored */

    case 16:/* Normal Interrupt Vector and Status register */

    case 17:/* Fast Interrupt vector and status register */

    case 18:/* Interrupt source register high */

    case 19:/* Interrupt source register low */

        return;



    case 20:/* Interrupt Force Register high */

        s->pending = (s->pending & 0xffffffffULL) | (val << 32);

        break;



    case 21:/* Interrupt Force Register low */

        s->pending = (s->pending & 0xffffffff00000000ULL) | val;

        break;



    case 22:/* Normal Interrupt Pending Register High */

    case 23:/* Normal Interrupt Pending Register Low */

    case 24: /* Fast Interrupt Pending Register High  */

    case 25: /* Fast Interrupt Pending Register Low  */

        return;



    default:

        IPRINTF("imx_avic_write: Bad offset %x\n", (int)offset);

    }

    imx_avic_update(s);

}
