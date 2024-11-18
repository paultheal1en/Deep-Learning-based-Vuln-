static void kvm_arm_gic_get(GICState *s)

{

    uint32_t reg;

    int i;

    int cpu;



    if (!kvm_arm_gic_can_save_restore(s)) {

            DPRINTF("Cannot get kernel gic state, no kernel interface");

            return;

    }



    /*****************************************************************

     * Distributor State

     */



    /* GICD_CTLR -> s->enabled */

    kvm_gicd_access(s, 0x0, 0, &reg, false);

    s->enabled = reg & 1;



    /* Sanity checking on GICD_TYPER -> s->num_irq, s->num_cpu */

    kvm_gicd_access(s, 0x4, 0, &reg, false);

    s->num_irq = ((reg & 0x1f) + 1) * 32;

    s->num_cpu = ((reg & 0xe0) >> 5) + 1;



    if (s->num_irq > GIC_MAXIRQ) {

            fprintf(stderr, "Too many IRQs reported from the kernel: %d\n",

                    s->num_irq);

            abort();

    }



    /* GICD_IIDR -> ? */

    kvm_gicd_access(s, 0x8, 0, &reg, false);



    /* Verify no GROUP 1 interrupts configured in the kernel */

    for_each_irq_reg(i, s->num_irq, 1) {

        kvm_gicd_access(s, 0x80 + (i * 4), 0, &reg, false);

        if (reg != 0) {

            fprintf(stderr, "Unsupported GICD_IGROUPRn value: %08x\n",

                    reg);

            abort();

        }

    }



    /* Clear all the IRQ settings */

    for (i = 0; i < s->num_irq; i++) {

        memset(&s->irq_state[i], 0, sizeof(s->irq_state[0]));

    }



    /* GICD_ISENABLERn -> irq_state[n].enabled */

    kvm_dist_get(s, 0x100, 1, s->num_irq, translate_enabled);



    /* GICD_ISPENDRn -> irq_state[n].pending + irq_state[n].level */

    kvm_dist_get(s, 0x200, 1, s->num_irq, translate_pending);



    /* GICD_ISACTIVERn -> irq_state[n].active */

    kvm_dist_get(s, 0x300, 1, s->num_irq, translate_active);



    /* GICD_ICFRn -> irq_state[n].trigger */

    kvm_dist_get(s, 0xc00, 2, s->num_irq, translate_trigger);



    /* GICD_IPRIORITYRn -> s->priorityX[irq] */

    kvm_dist_get(s, 0x400, 8, s->num_irq, translate_priority);



    /* GICD_ITARGETSRn -> s->irq_target[irq] */

    kvm_dist_get(s, 0x800, 8, s->num_irq, translate_targets);



    /* GICD_CPENDSGIRn -> s->sgi_pending */

    kvm_dist_get(s, 0xf10, 8, GIC_NR_SGIS, translate_sgisource);





    /*****************************************************************

     * CPU Interface(s) State

     */



    for (cpu = 0; cpu < s->num_cpu; cpu++) {

        /* GICC_CTLR -> s->cpu_enabled[cpu] */

        kvm_gicc_access(s, 0x00, cpu, &reg, false);

        s->cpu_enabled[cpu] = (reg & 1);



        /* GICC_PMR -> s->priority_mask[cpu] */

        kvm_gicc_access(s, 0x04, cpu, &reg, false);

        s->priority_mask[cpu] = (reg & 0xff);



        /* GICC_BPR -> s->bpr[cpu] */

        kvm_gicc_access(s, 0x08, cpu, &reg, false);

        s->bpr[cpu] = (reg & 0x7);



        /* GICC_ABPR -> s->abpr[cpu] */

        kvm_gicc_access(s, 0x1c, cpu, &reg, false);

        s->abpr[cpu] = (reg & 0x7);



        /* GICC_APRn -> s->apr[n][cpu] */

        for (i = 0; i < 4; i++) {

            kvm_gicc_access(s, 0xd0 + i * 4, cpu, &reg, false);

            s->apr[i][cpu] = reg;

        }

    }

}
