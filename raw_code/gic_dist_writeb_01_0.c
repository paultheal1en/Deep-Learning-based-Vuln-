static void gic_dist_writeb(void *opaque, target_phys_addr_t offset,

                            uint32_t value)

{

    GICState *s = (GICState *)opaque;

    int irq;

    int i;

    int cpu;



    cpu = gic_get_current_cpu(s);

    if (offset < 0x100) {

        if (offset == 0) {

            s->enabled = (value & 1);

            DPRINTF("Distribution %sabled\n", s->enabled ? "En" : "Dis");

        } else if (offset < 4) {

            /* ignored.  */

        } else if (offset >= 0x80) {

            /* Interrupt Security Registers, RAZ/WI */

        } else {

            goto bad_reg;

        }

    } else if (offset < 0x180) {

        /* Interrupt Set Enable.  */

        irq = (offset - 0x100) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < 16)

          value = 0xff;

        for (i = 0; i < 8; i++) {

            if (value & (1 << i)) {

                int mask = (irq < GIC_INTERNAL) ? (1 << cpu) : GIC_TARGET(irq);

                int cm = (irq < GIC_INTERNAL) ? (1 << cpu) : ALL_CPU_MASK;



                if (!GIC_TEST_ENABLED(irq + i, cm)) {

                    DPRINTF("Enabled IRQ %d\n", irq + i);

                }

                GIC_SET_ENABLED(irq + i, cm);

                /* If a raised level triggered IRQ enabled then mark

                   is as pending.  */

                if (GIC_TEST_LEVEL(irq + i, mask)

                        && !GIC_TEST_TRIGGER(irq + i)) {

                    DPRINTF("Set %d pending mask %x\n", irq + i, mask);

                    GIC_SET_PENDING(irq + i, mask);

                }

            }

        }

    } else if (offset < 0x200) {

        /* Interrupt Clear Enable.  */

        irq = (offset - 0x180) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < 16)

          value = 0;

        for (i = 0; i < 8; i++) {

            if (value & (1 << i)) {

                int cm = (irq < GIC_INTERNAL) ? (1 << cpu) : ALL_CPU_MASK;



                if (GIC_TEST_ENABLED(irq + i, cm)) {

                    DPRINTF("Disabled IRQ %d\n", irq + i);

                }

                GIC_CLEAR_ENABLED(irq + i, cm);

            }

        }

    } else if (offset < 0x280) {

        /* Interrupt Set Pending.  */

        irq = (offset - 0x200) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < 16)

          irq = 0;



        for (i = 0; i < 8; i++) {

            if (value & (1 << i)) {

                GIC_SET_PENDING(irq + i, GIC_TARGET(irq));

            }

        }

    } else if (offset < 0x300) {

        /* Interrupt Clear Pending.  */

        irq = (offset - 0x280) * 8 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        for (i = 0; i < 8; i++) {

            /* ??? This currently clears the pending bit for all CPUs, even

               for per-CPU interrupts.  It's unclear whether this is the

               corect behavior.  */

            if (value & (1 << i)) {

                GIC_CLEAR_PENDING(irq + i, ALL_CPU_MASK);

            }

        }

    } else if (offset < 0x400) {

        /* Interrupt Active.  */

        goto bad_reg;

    } else if (offset < 0x800) {

        /* Interrupt Priority.  */

        irq = (offset - 0x400) + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < GIC_INTERNAL) {

            s->priority1[irq][cpu] = value;

        } else {

            s->priority2[irq - GIC_INTERNAL] = value;

        }

    } else if (offset < 0xc00) {

        /* Interrupt CPU Target. RAZ/WI on uniprocessor GICs, with the

         * annoying exception of the 11MPCore's GIC.

         */

        if (s->num_cpu != 1 || s->revision == REV_11MPCORE) {

            irq = (offset - 0x800) + GIC_BASE_IRQ;

            if (irq >= s->num_irq) {

                goto bad_reg;

            }

            if (irq < 29) {

                value = 0;

            } else if (irq < GIC_INTERNAL) {

                value = ALL_CPU_MASK;

            }

            s->irq_target[irq] = value & ALL_CPU_MASK;

        }

    } else if (offset < 0xf00) {

        /* Interrupt Configuration.  */

        irq = (offset - 0xc00) * 4 + GIC_BASE_IRQ;

        if (irq >= s->num_irq)

            goto bad_reg;

        if (irq < GIC_INTERNAL)

            value |= 0xaa;

        for (i = 0; i < 4; i++) {

            if (value & (1 << (i * 2))) {

                GIC_SET_MODEL(irq + i);

            } else {

                GIC_CLEAR_MODEL(irq + i);

            }

            if (value & (2 << (i * 2))) {

                GIC_SET_TRIGGER(irq + i);

            } else {

                GIC_CLEAR_TRIGGER(irq + i);

            }

        }

    } else {

        /* 0xf00 is only handled for 32-bit writes.  */

        goto bad_reg;

    }

    gic_update(s);

    return;

bad_reg:

    hw_error("gic_dist_writeb: Bad offset %x\n", (int)offset);

}
