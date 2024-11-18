qemu_irq *openpic_init (MemoryRegion **pmem, int nb_cpus,

                        qemu_irq **irqs, qemu_irq irq_out)

{

    openpic_t *opp;

    int i, m;

    struct {

        const char             *name;

        MemoryRegionOps const  *ops;

        hwaddr      start_addr;

        ram_addr_t              size;

    } const list[] = {

        {"glb", &openpic_glb_ops, OPENPIC_GLB_REG_START, OPENPIC_GLB_REG_SIZE},

        {"tmr", &openpic_tmr_ops, OPENPIC_TMR_REG_START, OPENPIC_TMR_REG_SIZE},

        {"src", &openpic_src_ops, OPENPIC_SRC_REG_START, OPENPIC_SRC_REG_SIZE},

        {"cpu", &openpic_cpu_ops, OPENPIC_CPU_REG_START, OPENPIC_CPU_REG_SIZE},

    };



    /* XXX: for now, only one CPU is supported */

    if (nb_cpus != 1)

        return NULL;

    opp = g_malloc0(sizeof(openpic_t));



    memory_region_init(&opp->mem, "openpic", 0x40000);



    for (i = 0; i < ARRAY_SIZE(list); i++) {



        memory_region_init_io(&opp->sub_io_mem[i], list[i].ops, opp,

                              list[i].name, list[i].size);



        memory_region_add_subregion(&opp->mem, list[i].start_addr,

                                    &opp->sub_io_mem[i]);

    }



    //    isu_base &= 0xFFFC0000;

    opp->nb_cpus = nb_cpus;

    opp->max_irq = OPENPIC_MAX_IRQ;

    opp->irq_ipi0 = OPENPIC_IRQ_IPI0;

    opp->irq_tim0 = OPENPIC_IRQ_TIM0;

    /* Set IRQ types */

    for (i = 0; i < OPENPIC_EXT_IRQ; i++) {

        opp->src[i].type = IRQ_EXTERNAL;

    }

    for (; i < OPENPIC_IRQ_TIM0; i++) {

        opp->src[i].type = IRQ_SPECIAL;

    }

    m = OPENPIC_IRQ_IPI0;

    for (; i < m; i++) {

        opp->src[i].type = IRQ_TIMER;

    }

    for (; i < OPENPIC_MAX_IRQ; i++) {

        opp->src[i].type = IRQ_INTERNAL;

    }

    for (i = 0; i < nb_cpus; i++)

        opp->dst[i].irqs = irqs[i];

    opp->irq_out = irq_out;



    register_savevm(&opp->pci_dev.qdev, "openpic", 0, 2,

                    openpic_save, openpic_load, opp);

    qemu_register_reset(openpic_reset, opp);



    opp->irq_raise = openpic_irq_raise;

    opp->reset = openpic_reset;



    if (pmem)

        *pmem = &opp->mem;



    return qemu_allocate_irqs(openpic_set_irq, opp, opp->max_irq);

}
