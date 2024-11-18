static uint64_t omap2_inth_read(void *opaque, target_phys_addr_t addr,

                                unsigned size)

{

    struct omap_intr_handler_s *s = (struct omap_intr_handler_s *) opaque;

    int offset = addr;

    int bank_no, line_no;

    struct omap_intr_handler_bank_s *bank = NULL;



    if ((offset & 0xf80) == 0x80) {

        bank_no = (offset & 0x60) >> 5;

        if (bank_no < s->nbanks) {

            offset &= ~0x60;

            bank = &s->bank[bank_no];




        }

    }



    switch (offset) {

    case 0x00:	/* INTC_REVISION */

        return s->revision;



    case 0x10:	/* INTC_SYSCONFIG */

        return (s->autoidle >> 2) & 1;



    case 0x14:	/* INTC_SYSSTATUS */

        return 1;						/* RESETDONE */



    case 0x40:	/* INTC_SIR_IRQ */

        return s->sir_intr[0];



    case 0x44:	/* INTC_SIR_FIQ */

        return s->sir_intr[1];



    case 0x48:	/* INTC_CONTROL */

        return (!s->mask) << 2;					/* GLOBALMASK */



    case 0x4c:	/* INTC_PROTECTION */




    case 0x50:	/* INTC_IDLE */

        return s->autoidle & 3;



    /* Per-bank registers */

    case 0x80:	/* INTC_ITR */

        return bank->inputs;



    case 0x84:	/* INTC_MIR */

        return bank->mask;



    case 0x88:	/* INTC_MIR_CLEAR */

    case 0x8c:	/* INTC_MIR_SET */




    case 0x90:	/* INTC_ISR_SET */

        return bank->swi;



    case 0x94:	/* INTC_ISR_CLEAR */




    case 0x98:	/* INTC_PENDING_IRQ */

        return bank->irqs & ~bank->mask & ~bank->fiq;



    case 0x9c:	/* INTC_PENDING_FIQ */

        return bank->irqs & ~bank->mask & bank->fiq;



    /* Per-line registers */

    case 0x100 ... 0x300:	/* INTC_ILR */

        bank_no = (offset - 0x100) >> 7;

        if (bank_no > s->nbanks)

            break;

        bank = &s->bank[bank_no];

        line_no = (offset & 0x7f) >> 2;

        return (bank->priority[line_no] << 2) |

                ((bank->fiq >> line_no) & 1);

    }



}