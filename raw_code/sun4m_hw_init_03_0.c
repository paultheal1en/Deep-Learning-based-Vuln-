static void sun4m_hw_init(const struct hwdef *hwdef, int ram_size,

                          DisplayState *ds, const char *cpu_model)



{

    CPUState *env, *envs[MAX_CPUS];

    unsigned int i;

    void *iommu, *espdma, *ledma, *main_esp;

    const sparc_def_t *def;

    qemu_irq *cpu_irqs[MAX_CPUS], *slavio_irq, *slavio_cpu_irq,

        *espdma_irq, *ledma_irq;



    /* init CPUs */

    sparc_find_by_name(cpu_model, &def);

    if (def == NULL) {

        fprintf(stderr, "Unable to find Sparc CPU definition\n");

        exit(1);

    }



    for(i = 0; i < smp_cpus; i++) {

        env = cpu_init();

        cpu_sparc_register(env, def);

        envs[i] = env;

        if (i == 0) {

            qemu_register_reset(main_cpu_reset, env);

        } else {

            qemu_register_reset(secondary_cpu_reset, env);

            env->halted = 1;

        }

        register_savevm("cpu", i, 3, cpu_save, cpu_load, env);

        cpu_irqs[i] = qemu_allocate_irqs(cpu_set_irq, envs[i], MAX_PILS);

    }



    for (i = smp_cpus; i < MAX_CPUS; i++)

        cpu_irqs[i] = qemu_allocate_irqs(dummy_cpu_set_irq, NULL, MAX_PILS);



    /* allocate RAM */

    cpu_register_physical_memory(0, ram_size, 0);



    iommu = iommu_init(hwdef->iommu_base);

    slavio_intctl = slavio_intctl_init(hwdef->intctl_base,

                                       hwdef->intctl_base + 0x10000ULL,

                                       &hwdef->intbit_to_level[0],

                                       &slavio_irq, &slavio_cpu_irq,

                                       cpu_irqs,

                                       hwdef->clock_irq);



    espdma = sparc32_dma_init(hwdef->dma_base, slavio_irq[hwdef->esp_irq],

                              iommu, &espdma_irq);

    ledma = sparc32_dma_init(hwdef->dma_base + 16ULL,

                             slavio_irq[hwdef->le_irq], iommu, &ledma_irq);



    if (graphic_depth != 8 && graphic_depth != 24) {

        fprintf(stderr, "qemu: Unsupported depth: %d\n", graphic_depth);

        exit (1);

    }

    tcx_init(ds, hwdef->tcx_base, phys_ram_base + ram_size, ram_size,

             hwdef->vram_size, graphic_width, graphic_height, graphic_depth);



    if (nd_table[0].model == NULL

        || strcmp(nd_table[0].model, "lance") == 0) {

        lance_init(&nd_table[0], hwdef->le_base, ledma, *ledma_irq);

    } else if (strcmp(nd_table[0].model, "?") == 0) {

        fprintf(stderr, "qemu: Supported NICs: lance\n");

        exit (1);

    } else {

        fprintf(stderr, "qemu: Unsupported NIC: %s\n", nd_table[0].model);

        exit (1);

    }



    nvram = m48t59_init(slavio_irq[0], hwdef->nvram_base, 0,

                        hwdef->nvram_size, 8);

    for (i = 0; i < MAX_CPUS; i++) {

        slavio_timer_init(hwdef->counter_base +

                          (target_phys_addr_t)(i * TARGET_PAGE_SIZE),

                           slavio_cpu_irq[i], 0);

    }

    slavio_timer_init(hwdef->counter_base + 0x10000ULL,

                      slavio_irq[hwdef->clock1_irq], 2);

    slavio_serial_ms_kbd_init(hwdef->ms_kb_base, slavio_irq[hwdef->ms_kb_irq]);

    // Slavio TTYA (base+4, Linux ttyS0) is the first Qemu serial device

    // Slavio TTYB (base+0, Linux ttyS1) is the second Qemu serial device

    slavio_serial_init(hwdef->serial_base, slavio_irq[hwdef->ser_irq],

                       serial_hds[1], serial_hds[0]);

    fdctrl_init(slavio_irq[hwdef->fd_irq], 0, 1, hwdef->fd_base, fd_table);

    main_esp = esp_init(bs_table, hwdef->esp_base, espdma, *espdma_irq);



    for (i = 0; i < MAX_DISKS; i++) {

        if (bs_table[i]) {

            esp_scsi_attach(main_esp, bs_table[i], i);

        }

    }



    slavio_misc = slavio_misc_init(hwdef->slavio_base, hwdef->power_base,

                                   slavio_irq[hwdef->me_irq]);

    if (hwdef->cs_base != (target_phys_addr_t)-1)

        cs_init(hwdef->cs_base, hwdef->cs_irq, slavio_intctl);

}
