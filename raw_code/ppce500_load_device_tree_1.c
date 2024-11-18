static int ppce500_load_device_tree(MachineState *machine,

                                    PPCE500Params *params,

                                    hwaddr addr,

                                    hwaddr initrd_base,

                                    hwaddr initrd_size,

                                    hwaddr kernel_base,

                                    hwaddr kernel_size,

                                    bool dry_run)

{

    CPUPPCState *env = first_cpu->env_ptr;

    int ret = -1;

    uint64_t mem_reg_property[] = { 0, cpu_to_be64(machine->ram_size) };

    int fdt_size;

    void *fdt;

    uint8_t hypercall[16];

    uint32_t clock_freq = 400000000;

    uint32_t tb_freq = 400000000;

    int i;

    char compatible_sb[] = "fsl,mpc8544-immr\0simple-bus";

    char soc[128];

    char mpic[128];

    uint32_t mpic_ph;

    uint32_t msi_ph;

    char gutil[128];

    char pci[128];

    char msi[128];

    uint32_t *pci_map = NULL;

    int len;

    uint32_t pci_ranges[14] =

        {

            0x2000000, 0x0, params->pci_mmio_bus_base,

            params->pci_mmio_base >> 32, params->pci_mmio_base,

            0x0, 0x20000000,



            0x1000000, 0x0, 0x0,

            params->pci_pio_base >> 32, params->pci_pio_base,

            0x0, 0x10000,

        };

    QemuOpts *machine_opts = qemu_get_machine_opts();

    const char *dtb_file = qemu_opt_get(machine_opts, "dtb");

    const char *toplevel_compat = qemu_opt_get(machine_opts, "dt_compatible");



    if (dtb_file) {

        char *filename;

        filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, dtb_file);

        if (!filename) {

            goto out;

        }



        fdt = load_device_tree(filename, &fdt_size);


        if (!fdt) {

            goto out;

        }

        goto done;

    }



    fdt = create_device_tree(&fdt_size);

    if (fdt == NULL) {

        goto out;

    }



    /* Manipulate device tree in memory. */

    qemu_fdt_setprop_cell(fdt, "/", "#address-cells", 2);

    qemu_fdt_setprop_cell(fdt, "/", "#size-cells", 2);



    qemu_fdt_add_subnode(fdt, "/memory");

    qemu_fdt_setprop_string(fdt, "/memory", "device_type", "memory");

    qemu_fdt_setprop(fdt, "/memory", "reg", mem_reg_property,

                     sizeof(mem_reg_property));



    qemu_fdt_add_subnode(fdt, "/chosen");

    if (initrd_size) {

        ret = qemu_fdt_setprop_cell(fdt, "/chosen", "linux,initrd-start",

                                    initrd_base);

        if (ret < 0) {

            fprintf(stderr, "couldn't set /chosen/linux,initrd-start\n");

        }



        ret = qemu_fdt_setprop_cell(fdt, "/chosen", "linux,initrd-end",

                                    (initrd_base + initrd_size));

        if (ret < 0) {

            fprintf(stderr, "couldn't set /chosen/linux,initrd-end\n");

        }



    }



    if (kernel_base != -1ULL) {

        qemu_fdt_setprop_cells(fdt, "/chosen", "qemu,boot-kernel",

                                     kernel_base >> 32, kernel_base,

                                     kernel_size >> 32, kernel_size);

    }



    ret = qemu_fdt_setprop_string(fdt, "/chosen", "bootargs",

                                      machine->kernel_cmdline);

    if (ret < 0)

        fprintf(stderr, "couldn't set /chosen/bootargs\n");



    if (kvm_enabled()) {

        /* Read out host's frequencies */

        clock_freq = kvmppc_get_clockfreq();

        tb_freq = kvmppc_get_tbfreq();



        /* indicate KVM hypercall interface */

        qemu_fdt_add_subnode(fdt, "/hypervisor");

        qemu_fdt_setprop_string(fdt, "/hypervisor", "compatible",

                                "linux,kvm");

        kvmppc_get_hypercall(env, hypercall, sizeof(hypercall));

        qemu_fdt_setprop(fdt, "/hypervisor", "hcall-instructions",

                         hypercall, sizeof(hypercall));

        /* if KVM supports the idle hcall, set property indicating this */

        if (kvmppc_get_hasidle(env)) {

            qemu_fdt_setprop(fdt, "/hypervisor", "has-idle", NULL, 0);

        }

    }



    /* Create CPU nodes */

    qemu_fdt_add_subnode(fdt, "/cpus");

    qemu_fdt_setprop_cell(fdt, "/cpus", "#address-cells", 1);

    qemu_fdt_setprop_cell(fdt, "/cpus", "#size-cells", 0);



    /* We need to generate the cpu nodes in reverse order, so Linux can pick

       the first node as boot node and be happy */

    for (i = smp_cpus - 1; i >= 0; i--) {

        CPUState *cpu;

        PowerPCCPU *pcpu;

        char cpu_name[128];

        uint64_t cpu_release_addr = params->spin_base + (i * 0x20);



        cpu = qemu_get_cpu(i);

        if (cpu == NULL) {

            continue;

        }

        env = cpu->env_ptr;

        pcpu = POWERPC_CPU(cpu);



        snprintf(cpu_name, sizeof(cpu_name), "/cpus/PowerPC,8544@%x",

                 ppc_get_vcpu_dt_id(pcpu));

        qemu_fdt_add_subnode(fdt, cpu_name);

        qemu_fdt_setprop_cell(fdt, cpu_name, "clock-frequency", clock_freq);

        qemu_fdt_setprop_cell(fdt, cpu_name, "timebase-frequency", tb_freq);

        qemu_fdt_setprop_string(fdt, cpu_name, "device_type", "cpu");

        qemu_fdt_setprop_cell(fdt, cpu_name, "reg",

                              ppc_get_vcpu_dt_id(pcpu));

        qemu_fdt_setprop_cell(fdt, cpu_name, "d-cache-line-size",

                              env->dcache_line_size);

        qemu_fdt_setprop_cell(fdt, cpu_name, "i-cache-line-size",

                              env->icache_line_size);

        qemu_fdt_setprop_cell(fdt, cpu_name, "d-cache-size", 0x8000);

        qemu_fdt_setprop_cell(fdt, cpu_name, "i-cache-size", 0x8000);

        qemu_fdt_setprop_cell(fdt, cpu_name, "bus-frequency", 0);

        if (cpu->cpu_index) {

            qemu_fdt_setprop_string(fdt, cpu_name, "status", "disabled");

            qemu_fdt_setprop_string(fdt, cpu_name, "enable-method",

                                    "spin-table");

            qemu_fdt_setprop_u64(fdt, cpu_name, "cpu-release-addr",

                                 cpu_release_addr);

        } else {

            qemu_fdt_setprop_string(fdt, cpu_name, "status", "okay");

        }

    }



    qemu_fdt_add_subnode(fdt, "/aliases");

    /* XXX These should go into their respective devices' code */

    snprintf(soc, sizeof(soc), "/soc@%"PRIx64, params->ccsrbar_base);

    qemu_fdt_add_subnode(fdt, soc);

    qemu_fdt_setprop_string(fdt, soc, "device_type", "soc");

    qemu_fdt_setprop(fdt, soc, "compatible", compatible_sb,

                     sizeof(compatible_sb));

    qemu_fdt_setprop_cell(fdt, soc, "#address-cells", 1);

    qemu_fdt_setprop_cell(fdt, soc, "#size-cells", 1);

    qemu_fdt_setprop_cells(fdt, soc, "ranges", 0x0,

                           params->ccsrbar_base >> 32, params->ccsrbar_base,

                           MPC8544_CCSRBAR_SIZE);

    /* XXX should contain a reasonable value */

    qemu_fdt_setprop_cell(fdt, soc, "bus-frequency", 0);



    snprintf(mpic, sizeof(mpic), "%s/pic@%llx", soc, MPC8544_MPIC_REGS_OFFSET);

    qemu_fdt_add_subnode(fdt, mpic);

    qemu_fdt_setprop_string(fdt, mpic, "device_type", "open-pic");

    qemu_fdt_setprop_string(fdt, mpic, "compatible", "fsl,mpic");

    qemu_fdt_setprop_cells(fdt, mpic, "reg", MPC8544_MPIC_REGS_OFFSET,

                           0x40000);

    qemu_fdt_setprop_cell(fdt, mpic, "#address-cells", 0);

    qemu_fdt_setprop_cell(fdt, mpic, "#interrupt-cells", 2);

    mpic_ph = qemu_fdt_alloc_phandle(fdt);

    qemu_fdt_setprop_cell(fdt, mpic, "phandle", mpic_ph);

    qemu_fdt_setprop_cell(fdt, mpic, "linux,phandle", mpic_ph);

    qemu_fdt_setprop(fdt, mpic, "interrupt-controller", NULL, 0);



    /*

     * We have to generate ser1 first, because Linux takes the first

     * device it finds in the dt as serial output device. And we generate

     * devices in reverse order to the dt.

     */

    if (serial_hds[1]) {

        dt_serial_create(fdt, MPC8544_SERIAL1_REGS_OFFSET,

                         soc, mpic, "serial1", 1, false);

    }



    if (serial_hds[0]) {

        dt_serial_create(fdt, MPC8544_SERIAL0_REGS_OFFSET,

                         soc, mpic, "serial0", 0, true);

    }



    snprintf(gutil, sizeof(gutil), "%s/global-utilities@%llx", soc,

             MPC8544_UTIL_OFFSET);

    qemu_fdt_add_subnode(fdt, gutil);

    qemu_fdt_setprop_string(fdt, gutil, "compatible", "fsl,mpc8544-guts");

    qemu_fdt_setprop_cells(fdt, gutil, "reg", MPC8544_UTIL_OFFSET, 0x1000);

    qemu_fdt_setprop(fdt, gutil, "fsl,has-rstcr", NULL, 0);



    snprintf(msi, sizeof(msi), "/%s/msi@%llx", soc, MPC8544_MSI_REGS_OFFSET);

    qemu_fdt_add_subnode(fdt, msi);

    qemu_fdt_setprop_string(fdt, msi, "compatible", "fsl,mpic-msi");

    qemu_fdt_setprop_cells(fdt, msi, "reg", MPC8544_MSI_REGS_OFFSET, 0x200);

    msi_ph = qemu_fdt_alloc_phandle(fdt);

    qemu_fdt_setprop_cells(fdt, msi, "msi-available-ranges", 0x0, 0x100);

    qemu_fdt_setprop_phandle(fdt, msi, "interrupt-parent", mpic);

    qemu_fdt_setprop_cells(fdt, msi, "interrupts",

        0xe0, 0x0,

        0xe1, 0x0,

        0xe2, 0x0,

        0xe3, 0x0,

        0xe4, 0x0,

        0xe5, 0x0,

        0xe6, 0x0,

        0xe7, 0x0);

    qemu_fdt_setprop_cell(fdt, msi, "phandle", msi_ph);

    qemu_fdt_setprop_cell(fdt, msi, "linux,phandle", msi_ph);



    snprintf(pci, sizeof(pci), "/pci@%llx",

             params->ccsrbar_base + MPC8544_PCI_REGS_OFFSET);

    qemu_fdt_add_subnode(fdt, pci);

    qemu_fdt_setprop_cell(fdt, pci, "cell-index", 0);

    qemu_fdt_setprop_string(fdt, pci, "compatible", "fsl,mpc8540-pci");

    qemu_fdt_setprop_string(fdt, pci, "device_type", "pci");

    qemu_fdt_setprop_cells(fdt, pci, "interrupt-map-mask", 0xf800, 0x0,

                           0x0, 0x7);

    pci_map = pci_map_create(fdt, qemu_fdt_get_phandle(fdt, mpic),

                             params->pci_first_slot, params->pci_nr_slots,

                             &len);

    qemu_fdt_setprop(fdt, pci, "interrupt-map", pci_map, len);

    qemu_fdt_setprop_phandle(fdt, pci, "interrupt-parent", mpic);

    qemu_fdt_setprop_cells(fdt, pci, "interrupts", 24, 2);

    qemu_fdt_setprop_cells(fdt, pci, "bus-range", 0, 255);

    for (i = 0; i < 14; i++) {

        pci_ranges[i] = cpu_to_be32(pci_ranges[i]);

    }

    qemu_fdt_setprop_cell(fdt, pci, "fsl,msi", msi_ph);

    qemu_fdt_setprop(fdt, pci, "ranges", pci_ranges, sizeof(pci_ranges));

    qemu_fdt_setprop_cells(fdt, pci, "reg",

                           (params->ccsrbar_base + MPC8544_PCI_REGS_OFFSET) >> 32,

                           (params->ccsrbar_base + MPC8544_PCI_REGS_OFFSET),

                           0, 0x1000);

    qemu_fdt_setprop_cell(fdt, pci, "clock-frequency", 66666666);

    qemu_fdt_setprop_cell(fdt, pci, "#interrupt-cells", 1);

    qemu_fdt_setprop_cell(fdt, pci, "#size-cells", 2);

    qemu_fdt_setprop_cell(fdt, pci, "#address-cells", 3);

    qemu_fdt_setprop_string(fdt, "/aliases", "pci0", pci);



    if (params->has_mpc8xxx_gpio) {

        create_dt_mpc8xxx_gpio(fdt, soc, mpic);

    }



    if (params->has_platform_bus) {

        platform_bus_create_devtree(params, fdt, mpic);

    }



    params->fixup_devtree(params, fdt);



    if (toplevel_compat) {

        qemu_fdt_setprop(fdt, "/", "compatible", toplevel_compat,

                         strlen(toplevel_compat) + 1);

    }



done:

    if (!dry_run) {

        qemu_fdt_dumpdtb(fdt, fdt_size);

        cpu_physical_memory_write(addr, fdt, fdt_size);

    }

    ret = fdt_size;



out:

    g_free(pci_map);



    return ret;

}