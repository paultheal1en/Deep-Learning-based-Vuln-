void gic_update(GICState *s)

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

        for (irq = 0; irq < s->num_irq; irq++) {

            if (GIC_TEST_ENABLED(irq, cm) && gic_test_pending(s, irq, cm) &&

                (irq < GIC_INTERNAL || GIC_TARGET(irq) & cm)) {

                if (GIC_GET_PRIORITY(irq, cpu) < best_prio) {

                    best_prio = GIC_GET_PRIORITY(irq, cpu);

                    best_irq = irq;

                }

            }

        }

        level = 0;

        if (best_prio < s->priority_mask[cpu]) {

            s->current_pending[cpu] = best_irq;

            if (best_prio < s->running_priority[cpu]) {

                DPRINTF("Raised pending IRQ %d (cpu %d)\n", best_irq, cpu);

                level = 1;

            }

        }

        qemu_set_irq(s->parent_irq[cpu], level);

    }

}
