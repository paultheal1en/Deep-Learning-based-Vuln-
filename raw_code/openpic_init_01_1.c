static int openpic_init(SysBusDevice *dev)

{

    OpenPICState *opp = FROM_SYSBUS(typeof (*opp), dev);

    int i, j;

    struct memreg list_le[] = {

        {"glb", &openpic_glb_ops_le, true,

                OPENPIC_GLB_REG_START, OPENPIC_GLB_REG_SIZE},

        {"tmr", &openpic_tmr_ops_le, true,

                OPENPIC_TMR_REG_START, OPENPIC_TMR_REG_SIZE},

        {"msi", &openpic_msi_ops_le, true,

                OPENPIC_MSI_REG_START, OPENPIC_MSI_REG_SIZE},

        {"src", &openpic_src_ops_le, true,

                OPENPIC_SRC_REG_START, OPENPIC_SRC_REG_SIZE},

        {"cpu", &openpic_cpu_ops_le, true,

                OPENPIC_CPU_REG_START, OPENPIC_CPU_REG_SIZE},

    };

    struct memreg list_be[] = {

        {"glb", &openpic_glb_ops_be, true,

                OPENPIC_GLB_REG_START, OPENPIC_GLB_REG_SIZE},

        {"tmr", &openpic_tmr_ops_be, true,

                OPENPIC_TMR_REG_START, OPENPIC_TMR_REG_SIZE},

        {"msi", &openpic_msi_ops_be, true,

                OPENPIC_MSI_REG_START, OPENPIC_MSI_REG_SIZE},

        {"src", &openpic_src_ops_be, true,

                OPENPIC_SRC_REG_START, OPENPIC_SRC_REG_SIZE},

        {"cpu", &openpic_cpu_ops_be, true,

                OPENPIC_CPU_REG_START, OPENPIC_CPU_REG_SIZE},

    };

    struct memreg *list;



    switch (opp->model) {

    case OPENPIC_MODEL_FSL_MPIC_20:

    default:

        opp->flags |= OPENPIC_FLAG_IDE_CRIT;

        opp->nb_irqs = 80;

        opp->vid = VID_REVISION_1_2;

        opp->veni = VENI_GENERIC;

        opp->vector_mask = 0xFFFF;

        opp->tifr_reset = 0;

        opp->ipvp_reset = IPVP_MASK_MASK;

        opp->ide_reset = 1 << 0;

        opp->max_irq = FSL_MPIC_20_MAX_IRQ;

        opp->irq_ipi0 = FSL_MPIC_20_IPI_IRQ;

        opp->irq_tim0 = FSL_MPIC_20_TMR_IRQ;

        opp->irq_msi = FSL_MPIC_20_MSI_IRQ;

        opp->brr1 = FSL_BRR1_IPID | FSL_BRR1_IPMJ | FSL_BRR1_IPMN;

        msi_supported = true;

        list = list_be;

        break;

    case OPENPIC_MODEL_RAVEN:

        opp->nb_irqs = RAVEN_MAX_EXT;

        opp->vid = VID_REVISION_1_3;

        opp->veni = VENI_GENERIC;

        opp->vector_mask = 0xFF;

        opp->tifr_reset = 4160000;

        opp->ipvp_reset = IPVP_MASK_MASK | IPVP_MODE_MASK;

        opp->ide_reset = 0;

        opp->max_irq = RAVEN_MAX_IRQ;

        opp->irq_ipi0 = RAVEN_IPI_IRQ;

        opp->irq_tim0 = RAVEN_TMR_IRQ;

        opp->brr1 = -1;

        list = list_le;

        /* Don't map MSI region */

        list[2].map = false;



        /* Only UP supported today */

        if (opp->nb_cpus != 1) {

            return -EINVAL;

        }

        break;

    }



    memory_region_init(&opp->mem, "openpic", 0x40000);



    for (i = 0; i < ARRAY_SIZE(list_le); i++) {

        if (!list[i].map) {

            continue;

        }



        memory_region_init_io(&opp->sub_io_mem[i], list[i].ops, opp,

                              list[i].name, list[i].size);



        memory_region_add_subregion(&opp->mem, list[i].start_addr,

                                    &opp->sub_io_mem[i]);

    }



    for (i = 0; i < opp->nb_cpus; i++) {

        opp->dst[i].irqs = g_new(qemu_irq, OPENPIC_OUTPUT_NB);

        for (j = 0; j < OPENPIC_OUTPUT_NB; j++) {

            sysbus_init_irq(dev, &opp->dst[i].irqs[j]);

        }

    }



    register_savevm(&opp->busdev.qdev, "openpic", 0, 2,

                    openpic_save, openpic_load, opp);



    sysbus_init_mmio(dev, &opp->mem);

    qdev_init_gpio_in(&dev->qdev, openpic_set_irq, opp->max_irq);



    return 0;

}
