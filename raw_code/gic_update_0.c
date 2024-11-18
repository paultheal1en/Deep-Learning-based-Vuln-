static void gic_update(gic_state *s)

{

    int best_irq;

    int best_prio;

    int irq;

    int level;

    int cpu;

    int cm;



    for (cpu = 0; cpu < NUM_CPU(s); cpu++) {

        cm = 1 << cpu;

        s->current_pending[cpu] = 1023;

        if (!s->enabled || !s->cpu_enabled[cpu]) {

	    qemu_irq_lower(s->parent_irq[cpu]);

            return;

        }

        best_prio = 0x100;

        best_irq = 1023;

        for (irq = 0; irq < GIC_NIRQ; irq++) {

            if (GIC_TEST_ENABLED(irq) && GIC_TEST_PENDING(irq, cm)) {

                if (GIC_GET_PRIORITY(irq, cpu) < best_prio) {

                    best_prio = GIC_GET_PRIORITY(irq, cpu);

                    best_irq = irq;

                }

            }

        }

        level = 0;

        if (best_prio <= s->priority_mask[cpu]) {

            s->current_pending[cpu] = best_irq;

            if (best_prio < s->running_priority[cpu]) {

                DPRINTF("Raised pending IRQ %d\n", best_irq);

                level = 1;

            }

        }

        qemu_set_irq(s->parent_irq[cpu], level);

    }

}
