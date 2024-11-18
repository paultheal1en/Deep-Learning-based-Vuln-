qemu_irq *mpic_init (MemoryRegion *address_space, hwaddr base,

                     int nb_cpus, qemu_irq **irqs, qemu_irq irq_out)

{

    OpenPICState    *mpp;

    int           i;

    struct {

        const char             *name;

        MemoryRegionOps const  *ops;

        hwaddr      start_addr;

        ram_addr_t              size;

    } const list[] = {

        {"glb", &openpic_glb_ops_be, MPIC_GLB_REG_START, MPIC_GLB_REG_SIZE},

        {"tmr", &openpic_tmr_ops_be, MPIC_TMR_REG_START, MPIC_TMR_REG_SIZE},

        {"src", &openpic_src_ops_be, MPIC_SRC_REG_START, MPIC_SRC_REG_SIZE},

        {"cpu", &openpic_cpu_ops_be, MPIC_CPU_REG_START, MPIC_CPU_REG_SIZE},

    };



    mpp = g_malloc0(sizeof(OpenPICState));



    memory_region_init(&mpp->mem, "mpic", 0x40000);

    memory_region_add_subregion(address_space, base, &mpp->mem);



    for (i = 0; i < sizeof(list)/sizeof(list[0]); i++) {



        memory_region_init_io(&mpp->sub_io_mem[i], list[i].ops, mpp,

                              list[i].name, list[i].size);



        memory_region_add_subregion(&mpp->mem, list[i].start_addr,

                                    &mpp->sub_io_mem[i]);

    }



    mpp->nb_cpus = nb_cpus;

    /* 12 external sources, 48 internal sources , 4 timer sources,

       4 IPI sources, 4 messaging sources, and 8 Shared MSI sources */

    mpp->nb_irqs = 80;

    mpp->vid = VID_REVISION_1_2;

    mpp->veni = VENI_GENERIC;

    mpp->spve_mask = 0xFFFF;

    mpp->tifr_reset = 0x00000000;

    mpp->ipvp_reset = 0x80000000;

    mpp->ide_reset = 0x00000001;

    mpp->max_irq = MPIC_MAX_IRQ;

    mpp->irq_ipi0 = MPIC_IPI_IRQ;

    mpp->irq_tim0 = MPIC_TMR_IRQ;



    for (i = 0; i < nb_cpus; i++)

        mpp->dst[i].irqs = irqs[i];

    mpp->irq_out = irq_out;



    /* Enable critical interrupt support */

    mpp->flags |= OPENPIC_FLAG_IDE_CRIT;



    register_savevm(NULL, "mpic", 0, 2, openpic_save, openpic_load, mpp);

    qemu_register_reset(openpic_reset, mpp);



    return qemu_allocate_irqs(openpic_set_irq, mpp, mpp->max_irq);

}
