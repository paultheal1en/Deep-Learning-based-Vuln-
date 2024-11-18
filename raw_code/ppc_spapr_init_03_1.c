static void ppc_spapr_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    const char *boot_device = machine->boot_order;

    PowerPCCPU *cpu;

    CPUPPCState *env;

    PCIHostState *phb;

    int i;

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *rma_region;

    void *rma = NULL;

    hwaddr rma_alloc_size;

    hwaddr node0_size = spapr_node0_size();

    uint32_t initrd_base = 0;

    long kernel_size = 0, initrd_size = 0;

    long load_limit, rtas_limit, fw_size;

    bool kernel_le = false;

    char *filename;



    msi_supported = true;



    spapr = g_malloc0(sizeof(*spapr));

    QLIST_INIT(&spapr->phbs);



    cpu_ppc_hypercall = emulate_spapr_hypercall;



    /* Allocate RMA if necessary */

    rma_alloc_size = kvmppc_alloc_rma(&rma);



    if (rma_alloc_size == -1) {

        hw_error("qemu: Unable to create RMA\n");

        exit(1);

    }



    if (rma_alloc_size && (rma_alloc_size < node0_size)) {

        spapr->rma_size = rma_alloc_size;

    } else {

        spapr->rma_size = node0_size;



        /* With KVM, we don't actually know whether KVM supports an

         * unbounded RMA (PR KVM) or is limited by the hash table size

         * (HV KVM using VRMA), so we always assume the latter

         *

         * In that case, we also limit the initial allocations for RTAS

         * etc... to 256M since we have no way to know what the VRMA size

         * is going to be as it depends on the size of the hash table

         * isn't determined yet.

         */

        if (kvm_enabled()) {

            spapr->vrma_adjust = 1;

            spapr->rma_size = MIN(spapr->rma_size, 0x10000000);

        }

    }



    if (spapr->rma_size > node0_size) {

        fprintf(stderr, "Error: Numa node 0 has to span the RMA (%#08"HWADDR_PRIx")\n",

                spapr->rma_size);

        exit(1);

    }



    /* We place the device tree and RTAS just below either the top of the RMA,

     * or just below 2GB, whichever is lowere, so that it can be

     * processed with 32-bit real mode code if necessary */

    rtas_limit = MIN(spapr->rma_size, 0x80000000);

    spapr->rtas_addr = rtas_limit - RTAS_MAX_SIZE;

    spapr->fdt_addr = spapr->rtas_addr - FDT_MAX_SIZE;

    load_limit = spapr->fdt_addr - FW_OVERHEAD;



    /* We aim for a hash table of size 1/128 the size of RAM.  The

     * normal rule of thumb is 1/64 the size of RAM, but that's much

     * more than needed for the Linux guests we support. */

    spapr->htab_shift = 18; /* Minimum architected size */

    while (spapr->htab_shift <= 46) {

        if ((1ULL << (spapr->htab_shift + 7)) >= ram_size) {

            break;

        }

        spapr->htab_shift++;

    }



    /* Set up Interrupt Controller before we create the VCPUs */

    spapr->icp = xics_system_init(smp_cpus * kvmppc_smt_threads() / smp_threads,

                                  XICS_IRQS);



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



        /* PAPR always has exception vectors in RAM not ROM. To ensure this,

         * MSR[IP] should never be set.

         */

        env->msr_mask &= ~(1 << 6);



        /* Tell KVM that we're in PAPR mode */

        if (kvm_enabled()) {

            kvmppc_set_papr(cpu);

        }



        if (cpu->max_compat) {

            if (ppc_set_compat(cpu, cpu->max_compat) < 0) {

                exit(1);

            }

        }



        xics_cpu_setup(spapr->icp, cpu);



        qemu_register_reset(spapr_cpu_reset, cpu);

    }



    /* allocate RAM */

    spapr->ram_limit = ram_size;

    memory_region_allocate_system_memory(ram, NULL, "ppc_spapr.ram",

                                         spapr->ram_limit);

    memory_region_add_subregion(sysmem, 0, ram);



    if (rma_alloc_size && rma) {

        rma_region = g_new(MemoryRegion, 1);

        memory_region_init_ram_ptr(rma_region, NULL, "ppc_spapr.rma",

                                   rma_alloc_size, rma);

        vmstate_register_ram_global(rma_region);

        memory_region_add_subregion(sysmem, 0, rma_region);

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



    /* Set up EPOW events infrastructure */

    spapr_events_init(spapr);



    /* Set up VIO bus */

    spapr->vio_bus = spapr_vio_bus_init();



    for (i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            spapr_vty_create(spapr->vio_bus, serial_hds[i]);

        }

    }



    /* We always have at least the nvram device on VIO */

    spapr_create_nvram(spapr);



    /* Set up PCI */

    spapr_pci_msi_init(spapr, SPAPR_PCI_MSI_WINDOW);

    spapr_pci_rtas_init();



    phb = spapr_create_phb(spapr, 0);



    for (i = 0; i < nb_nics; i++) {

        NICInfo *nd = &nd_table[i];



        if (!nd->model) {

            nd->model = g_strdup("ibmveth");

        }



        if (strcmp(nd->model, "ibmveth") == 0) {

            spapr_vlan_create(spapr->vio_bus, nd);

        } else {

            pci_nic_init_nofail(&nd_table[i], phb->bus, nd->model, NULL);

        }

    }



    for (i = 0; i <= drive_get_max_bus(IF_SCSI); i++) {

        spapr_vscsi_create(spapr->vio_bus);

    }



    /* Graphics */

    if (spapr_vga_init(phb->bus)) {

        spapr->has_graphics = true;

    }



    if (usb_enabled(spapr->has_graphics)) {

        pci_create_simple(phb->bus, -1, "pci-ohci");

        if (spapr->has_graphics) {

            usbdevice_create("keyboard");

            usbdevice_create("mouse");

        }

    }



    if (spapr->rma_size < (MIN_RMA_SLOF << 20)) {

        fprintf(stderr, "qemu: pSeries SLOF firmware requires >= "

                "%ldM guest RMA (Real Mode Area memory)\n", MIN_RMA_SLOF);

        exit(1);

    }



    if (kernel_filename) {

        uint64_t lowaddr = 0;



        kernel_size = load_elf(kernel_filename, translate_kernel_address, NULL,

                               NULL, &lowaddr, NULL, 1, ELF_MACHINE, 0);

        if (kernel_size == ELF_LOAD_WRONG_ENDIAN) {

            kernel_size = load_elf(kernel_filename,

                                   translate_kernel_address, NULL,

                                   NULL, &lowaddr, NULL, 0, ELF_MACHINE, 0);

            kernel_le = kernel_size > 0;

        }

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: error loading %s: %s\n",

                    kernel_filename, load_elf_strerror(kernel_size));

            exit(1);

        }



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

        } else {

            initrd_base = 0;

            initrd_size = 0;

        }

    }



    if (bios_name == NULL) {

        bios_name = FW_FILE_NAME;

    }

    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

    fw_size = load_image_targphys(filename, 0, FW_MAX_SIZE);

    if (fw_size < 0) {

        hw_error("qemu: could not load LPAR rtas '%s'\n", filename);

        exit(1);

    }

    g_free(filename);



    spapr->entry_point = 0x100;



    vmstate_register(NULL, 0, &vmstate_spapr, spapr);

    register_savevm_live(NULL, "spapr/htab", -1, 1,

                         &savevm_htab_handlers, spapr);



    /* Prepare the device tree */

    spapr->fdt_skel = spapr_create_fdt_skel(initrd_base, initrd_size,

                                            kernel_size, kernel_le,

                                            boot_device, kernel_cmdline,

                                            spapr->epow_irq);

    assert(spapr->fdt_skel != NULL);

}
