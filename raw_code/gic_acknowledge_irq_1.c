uint32_t gic_acknowledge_irq(GICState *s, int cpu, MemTxAttrs attrs)

{

    int ret, irq, src;

    int cm = 1 << cpu;



    /* gic_get_current_pending_irq() will return 1022 or 1023 appropriately

     * for the case where this GIC supports grouping and the pending interrupt

     * is in the wrong group.

     */

    irq = gic_get_current_pending_irq(s, cpu, attrs);




    if (irq >= GIC_MAXIRQ) {

        DPRINTF("ACK, no pending interrupt or it is hidden: %d\n", irq);

        return irq;

    }



    if (GIC_GET_PRIORITY(irq, cpu) >= s->running_priority[cpu]) {

        DPRINTF("ACK, pending interrupt (%d) has insufficient priority\n", irq);

        return 1023;

    }



    if (s->revision == REV_11MPCORE || s->revision == REV_NVIC) {

        /* Clear pending flags for both level and edge triggered interrupts.

         * Level triggered IRQs will be reasserted once they become inactive.

         */

        GIC_CLEAR_PENDING(irq, GIC_TEST_MODEL(irq) ? ALL_CPU_MASK : cm);

        ret = irq;

    } else {

        if (irq < GIC_NR_SGIS) {

            /* Lookup the source CPU for the SGI and clear this in the

             * sgi_pending map.  Return the src and clear the overall pending

             * state on this CPU if the SGI is not pending from any CPUs.

             */

            assert(s->sgi_pending[irq][cpu] != 0);

            src = ctz32(s->sgi_pending[irq][cpu]);

            s->sgi_pending[irq][cpu] &= ~(1 << src);

            if (s->sgi_pending[irq][cpu] == 0) {

                GIC_CLEAR_PENDING(irq, GIC_TEST_MODEL(irq) ? ALL_CPU_MASK : cm);

            }

            ret = irq | ((src & 0x7) << 10);

        } else {

            /* Clear pending state for both level and edge triggered

             * interrupts. (level triggered interrupts with an active line

             * remain pending, see gic_test_pending)

             */

            GIC_CLEAR_PENDING(irq, GIC_TEST_MODEL(irq) ? ALL_CPU_MASK : cm);

            ret = irq;

        }

    }



    gic_activate_irq(s, cpu, irq);

    gic_update(s);

    DPRINTF("ACK %d\n", irq);

    return ret;

}