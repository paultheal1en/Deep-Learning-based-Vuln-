static uint32_t gic_dist_readb(void *opaque, hwaddr offset, MemTxAttrs attrs)

{

    GICState *s = (GICState *)opaque;

    uint32_t res;

    int irq;

    int i;

    int cpu;

    int cm;

    int mask;



    cpu = gic_get_current_cpu(s);

    cm = 1 << cpu;

    if (offset < 0x100) {

        if (offset == 0)

            return s->enabled;

        if (offset == 4)

            /* Interrupt Controller Type Register */

            return ((s->num_irq / 32) - 1)

                    | ((NUM_CPU(s) - 1) << 5)

                    | (s->security_extn << 10);

        if (offset < 0x08)

            return 0;

        if (offset >= 0x80) {

            /* Interrupt Group Registers: these RAZ/WI if this is an NS

             * access to a GIC with the security extensions, or if the GIC

             * doesn't have groups at all.

             */

            res = 0;

            if (!(s->security_extn && !attrs.secure) && gic_has_groups(s)) {

                /* Every byte offset holds 8 group status bits */

                irq = (offset - 0x080) * 8 + GIC_BASE_IRQ;

                if (irq >= s->num_irq) {

                    goto bad_reg;

                }

                for (i = 0; i < 8; i++) {

                    if (GIC_TEST_GROUP(irq + i, cm)) {

                        res |= (1 << i);

                    }

                }

            }

            return res;

        }

        goto bad_reg;

    } else if (offset < 0x200) {

        /* Interrupt Set/Clear Enable.  */

        if (offset < 0x180)

            irq = (offset - 0x100) * 8;

        else

            irq = (offset - 0x180) * 8;

        irq += GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        res = 0;

        for (i = 0; i < 8; i++) {

            if (GIC_TEST_ENABLED(irq + i, cm)) {

                res |= (1 << i);

            }

        }

    } else if (offset < 0x300) {

        /* Interrupt Set/Clear Pending.  */

        if (offset < 0x280)

            irq = (offset - 0x200) * 8;

        else

            irq = (offset - 0x280) * 8;

        irq += GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        res = 0;

        mask = (irq < GIC_INTERNAL) ?  cm : ALL_CPU_MASK;

        for (i = 0; i < 8; i++) {

            if (gic_test_pending(s, irq + i, mask)) {

                res |= (1 << i);

            }

        }

    } else if (offset < 0x400) {

        /* Interrupt Active.  */

        irq = (offset - 0x300) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        res = 0;

        mask = (irq < GIC_INTERNAL) ?  cm : ALL_CPU_MASK;

        for (i = 0; i < 8; i++) {

            if (GIC_TEST_ACTIVE(irq + i, mask)) {

                res |= (1 << i);

            }

        }

    } else if (offset < 0x800) {

        /* Interrupt Priority.  */

        irq = (offset - 0x400) + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        res = GIC_GET_PRIORITY(irq, cpu);

    } else if (offset < 0xc00) {

        /* Interrupt CPU Target.  */

        if (s->num_cpu == 1 && s->revision != REV_11MPCORE) {

            /* For uniprocessor GICs these RAZ/WI */

            res = 0;

        } else {

            irq = (offset - 0x800) + GIC_BASE_IRQ;

            if (irq >= s->num_irq) {

                goto bad_reg;

            }

            if (irq >= 29 && irq <= 31) {

                res = cm;

            } else {

                res = GIC_TARGET(irq);

            }

        }

    } else if (offset < 0xf00) {

        /* Interrupt Configuration.  */

        irq = (offset - 0xc00) * 4 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        res = 0;

        for (i = 0; i < 4; i++) {

            if (GIC_TEST_MODEL(irq + i))

                res |= (1 << (i * 2));

            if (GIC_TEST_EDGE_TRIGGER(irq + i))

                res |= (2 << (i * 2));

        }

    } else if (offset < 0xf10) {

        goto bad_reg;

    } else if (offset < 0xf30) {

        if (s->revision == REV_11MPCORE || s->revision == REV_NVIC) {

            goto bad_reg;

        }



        if (offset < 0xf20) {

            /* GICD_CPENDSGIRn */

            irq = (offset - 0xf10);

        } else {

            irq = (offset - 0xf20);

            /* GICD_SPENDSGIRn */

        }



        res = s->sgi_pending[irq][cpu];

    } else if (offset < 0xfe0) {

        goto bad_reg;

    } else /* offset >= 0xfe0 */ {

        if (offset & 3) {

            res = 0;

        } else {

            res = gic_id[(offset - 0xfe0) >> 2];

        }

    }

    return res;

bad_reg:

    qemu_log_mask(LOG_GUEST_ERROR,

                  "gic_dist_readb: Bad offset %x\n", (int)offset);

    return 0;

}
