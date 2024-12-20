static void ppc_spapr_init(ram_addr_t ram_size,
                           const char *boot_device,
                           const char *kernel_filename,
                           const char *kernel_cmdline,
                           const char *initrd_filename,
                           const char *cpu_model)
{
    PowerPCCPU *cpu;
    CPUPPCState *env;
    int i;
    MemoryRegion *sysmem = get_system_memory();
    MemoryRegion *ram = g_new(MemoryRegion, 1);
    target_phys_addr_t rma_alloc_size, rma_size;
    uint32_t initrd_base = 0;
    long kernel_size = 0, initrd_size = 0;
    long load_limit, rtas_limit, fw_size;
    long pteg_shift = 17;
    char *filename;
    spapr = g_malloc0(sizeof(*spapr));
    QLIST_INIT(&spapr->phbs);
    cpu_ppc_hypercall = emulate_spapr_hypercall;
    /* Allocate RMA if necessary */
    rma_alloc_size = kvmppc_alloc_rma("ppc_spapr.rma", sysmem);
    if (rma_alloc_size == -1) {
        hw_error("qemu: Unable to create RMA\n");
        exit(1);
    }
    if (rma_alloc_size && (rma_alloc_size < ram_size)) {
        rma_size = rma_alloc_size;
    } else {
        rma_size = ram_size;
    }
    /* We place the device tree and RTAS just below either the top of the RMA,
     * or just below 2GB, whichever is lowere, so that it can be
     * processed with 32-bit real mode code if necessary */
    rtas_limit = MIN(rma_size, 0x80000000);
    spapr->rtas_addr = rtas_limit - RTAS_MAX_SIZE;
    spapr->fdt_addr = spapr->rtas_addr - FDT_MAX_SIZE;
    load_limit = spapr->fdt_addr - FW_OVERHEAD;
    /* init CPUs */
    if (cpu_model == NULL) {
        cpu_model = kvm_enabled() ? "host" : "POWER7";
    }
    for (i = 0; i < smp_cpus; i++) {
        cpu = cpu_ppc_init(cpu_model);
        if (cpu == NULL) {
            fprintf(stderr, "Unable to find PowerPC CPU definition\n");
            exit(1);
        }
        env = &cpu->env;
        /* Set time-base frequency to 512 MHz */
        cpu_ppc_tb_init(env, TIMEBASE_FREQ);
        qemu_register_reset(spapr_cpu_reset, cpu);
        env->hreset_vector = 0x60;
        env->hreset_excp_prefix = 0;
        env->gpr[3] = env->cpu_index;
    }
    /* allocate RAM */
    spapr->ram_limit = ram_size;
    if (spapr->ram_limit > rma_alloc_size) {
        ram_addr_t nonrma_base = rma_alloc_size;
        ram_addr_t nonrma_size = spapr->ram_limit - rma_alloc_size;
        memory_region_init_ram(ram, "ppc_spapr.ram", nonrma_size);
        vmstate_register_ram_global(ram);
        memory_region_add_subregion(sysmem, nonrma_base, ram);
    }
    /* allocate hash page table.  For now we always make this 16mb,
     * later we should probably make it scale to the size of guest
     * RAM */
    spapr->htab_size = 1ULL << (pteg_shift + 7);
    spapr->htab = qemu_memalign(spapr->htab_size, spapr->htab_size);
    for (env = first_cpu; env != NULL; env = env->next_cpu) {
        env->external_htab = spapr->htab;
        env->htab_base = -1;
        env->htab_mask = spapr->htab_size - 1;
        /* Tell KVM that we're in PAPR mode */
        env->spr[SPR_SDR1] = (unsigned long)spapr->htab |
                             ((pteg_shift + 7) - 18);
        env->spr[SPR_HIOR] = 0;
        if (kvm_enabled()) {
            kvmppc_set_papr(env);
        }
    }
    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, "spapr-rtas.bin");
    spapr->rtas_size = load_image_targphys(filename, spapr->rtas_addr,
                                           rtas_limit - spapr->rtas_addr);
    if (spapr->rtas_size < 0) {
        hw_error("qemu: could not load LPAR rtas '%s'\n", filename);
        exit(1);
    }
    if (spapr->rtas_size > RTAS_MAX_SIZE) {
        hw_error("RTAS too big ! 0x%lx bytes (max is 0x%x)\n",
                 spapr->rtas_size, RTAS_MAX_SIZE);
        exit(1);
    }
    g_free(filename);
    /* Set up Interrupt Controller */
    spapr->icp = xics_system_init(XICS_IRQS);
    spapr->next_irq = 16;
    /* Set up VIO bus */
    spapr->vio_bus = spapr_vio_bus_init();
    for (i = 0; i < MAX_SERIAL_PORTS; i++) {
        if (serial_hds[i]) {
            spapr_vty_create(spapr->vio_bus, serial_hds[i]);
        }
    }
    /* Set up PCI */
    spapr_create_phb(spapr, "pci", SPAPR_PCI_BUID,
                     SPAPR_PCI_MEM_WIN_ADDR,
                     SPAPR_PCI_MEM_WIN_SIZE,
                     SPAPR_PCI_IO_WIN_ADDR);
    for (i = 0; i < nb_nics; i++) {
        NICInfo *nd = &nd_table[i];
        if (!nd->model) {
            nd->model = g_strdup("ibmveth");
        }
        if (strcmp(nd->model, "ibmveth") == 0) {
            spapr_vlan_create(spapr->vio_bus, nd);
        } else {
            pci_nic_init_nofail(&nd_table[i], nd->model, NULL);
        }
    }
    for (i = 0; i <= drive_get_max_bus(IF_SCSI); i++) {
        spapr_vscsi_create(spapr->vio_bus);
    }
    if (rma_size < (MIN_RMA_SLOF << 20)) {
        fprintf(stderr, "qemu: pSeries SLOF firmware requires >= "
                "%ldM guest RMA (Real Mode Area memory)\n", MIN_RMA_SLOF);
        exit(1);
    }
    fprintf(stderr, "sPAPR memory map:\n");
    fprintf(stderr, "RTAS                 : 0x%08lx..%08lx\n",
            (unsigned long)spapr->rtas_addr,
            (unsigned long)(spapr->rtas_addr + spapr->rtas_size - 1));
    fprintf(stderr, "FDT                  : 0x%08lx..%08lx\n",
            (unsigned long)spapr->fdt_addr,
            (unsigned long)(spapr->fdt_addr + FDT_MAX_SIZE - 1));
    if (kernel_filename) {
        uint64_t lowaddr = 0;
        kernel_size = load_elf(kernel_filename, translate_kernel_address, NULL,
                               NULL, &lowaddr, NULL, 1, ELF_MACHINE, 0);
        if (kernel_size < 0) {
            kernel_size = load_image_targphys(kernel_filename,
                                              KERNEL_LOAD_ADDR,
                                              load_limit - KERNEL_LOAD_ADDR);
        }
        if (kernel_size < 0) {
            fprintf(stderr, "qemu: could not load kernel '%s'\n",
                    kernel_filename);
            exit(1);
        }
        fprintf(stderr, "Kernel               : 0x%08x..%08lx\n",
                KERNEL_LOAD_ADDR, KERNEL_LOAD_ADDR + kernel_size - 1);
        /* load initrd */
        if (initrd_filename) {
            /* Try to locate the initrd in the gap between the kernel
             * and the firmware. Add a bit of space just in case
             */
            initrd_base = (KERNEL_LOAD_ADDR + kernel_size + 0x1ffff) & ~0xffff;
            initrd_size = load_image_targphys(initrd_filename, initrd_base,
                                              load_limit - initrd_base);
            if (initrd_size < 0) {
                fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",
                        initrd_filename);
                exit(1);
            }
            fprintf(stderr, "Ramdisk              : 0x%08lx..%08lx\n",
                    (long)initrd_base, (long)(initrd_base + initrd_size - 1));
        } else {
            initrd_base = 0;
            initrd_size = 0;
        }
    }
    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, FW_FILE_NAME);
    fw_size = load_image_targphys(filename, 0, FW_MAX_SIZE);
    if (fw_size < 0) {
        hw_error("qemu: could not load LPAR rtas '%s'\n", filename);
        exit(1);
    }
    g_free(filename);
    fprintf(stderr, "Firmware load        : 0x%08x..%08lx\n",
            0, fw_size);
    fprintf(stderr, "Firmware runtime     : 0x%08lx..%08lx\n",
            load_limit, (unsigned long)spapr->fdt_addr);
    spapr->entry_point = 0x100;
    /* SLOF will startup the secondary CPUs using RTAS */
    for (env = first_cpu; env != NULL; env = env->next_cpu) {
        env->halted = 1;
    }
    /* Prepare the device tree */
    spapr->fdt_skel = spapr_create_fdt_skel(cpu_model, rma_size,
                                            initrd_base, initrd_size,
                                            kernel_size,
                                            boot_device, kernel_cmdline,
                                            pteg_shift + 7);
    assert(spapr->fdt_skel != NULL);
    qemu_register_reset(spapr_reset, spapr);
}