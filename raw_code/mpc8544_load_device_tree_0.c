static int mpc8544_load_device_tree(CPUPPCState *env,

                                    target_phys_addr_t addr,

                                    target_phys_addr_t ramsize,

                                    target_phys_addr_t initrd_base,

                                    target_phys_addr_t initrd_size,

                                    const char *kernel_cmdline)

{

    int ret = -1;

    uint32_t mem_reg_property[] = {0, cpu_to_be32(ramsize)};

    int fdt_size;

    void *fdt;

    uint8_t hypercall[16];

    uint32_t clock_freq = 400000000;

    uint32_t tb_freq = 400000000;

    int i;

    char compatible[] = "MPC8544DS\0MPC85xxDS";

    char model[] = "MPC8544DS";

    char soc[128];

    char ser0[128];

    char ser1[128];

    char mpic[128];

    uint32_t mpic_ph;

    char gutil[128];

    char pci[128];

    uint32_t pci_map[7 * 8];

    uint32_t pci_ranges[12] = { 0x2000000, 0x0, 0xc0000000, 0xc0000000, 0x0,

                                0x20000000, 0x1000000, 0x0, 0x0, 0xe1000000,

                                0x0, 0x10000 };

    QemuOpts *machine_opts;

    const char *dumpdtb = NULL;



    fdt = create_device_tree(&fdt_size);

    if (fdt == NULL) {

        goto out;

    }



    /* Manipulate device tree in memory. */

    qemu_devtree_setprop_string(fdt, "/", "model", model);

    qemu_devtree_setprop(fdt, "/", "compatible", compatible,

                         sizeof(compatible));

    qemu_devtree_setprop_cell(fdt, "/", "#address-cells", 1);

    qemu_devtree_setprop_cell(fdt, "/", "#size-cells", 1);



    qemu_devtree_add_subnode(fdt, "/memory");

    qemu_devtree_setprop_string(fdt, "/memory", "device_type", "memory");

    qemu_devtree_setprop(fdt, "/memory", "reg", mem_reg_property,

                         sizeof(mem_reg_property));



    qemu_devtree_add_subnode(fdt, "/chosen");

    if (initrd_size) {

        ret = qemu_devtree_setprop_cell(fdt, "/chosen", "linux,initrd-start",

                                        initrd_base);

        if (ret < 0) {

            fprintf(stderr, "couldn't set /chosen/linux,initrd-start\n");

        }



        ret = qemu_devtree_setprop_cell(fdt, "/chosen", "linux,initrd-end",

                                        (initrd_base + initrd_size));

        if (ret < 0) {

            fprintf(stderr, "couldn't set /chosen/linux,initrd-end\n");

        }

    }



    ret = qemu_devtree_setprop_string(fdt, "/chosen", "bootargs",

                                      kernel_cmdline);

    if (ret < 0)

        fprintf(stderr, "couldn't set /chosen/bootargs\n");



    if (kvm_enabled()) {

        /* Read out host's frequencies */

        clock_freq = kvmppc_get_clockfreq();

        tb_freq = kvmppc_get_tbfreq();



        /* indicate KVM hypercall interface */

        qemu_devtree_add_subnode(fdt, "/hypervisor");

        qemu_devtree_setprop_string(fdt, "/hypervisor", "compatible",

                                    "linux,kvm");

        kvmppc_get_hypercall(env, hypercall, sizeof(hypercall));

        qemu_devtree_setprop(fdt, "/hypervisor", "hcall-instructions",

                             hypercall, sizeof(hypercall));

    }



    /* Create CPU nodes */

    qemu_devtree_add_subnode(fdt, "/cpus");

    qemu_devtree_setprop_cell(fdt, "/cpus", "#address-cells", 1);

    qemu_devtree_setprop_cell(fdt, "/cpus", "#size-cells", 0);



    /* We need to generate the cpu nodes in reverse order, so Linux can pick

       the first node as boot node and be happy */

    for (i = smp_cpus - 1; i >= 0; i--) {

        char cpu_name[128];

        uint64_t cpu_release_addr = MPC8544_SPIN_BASE + (i * 0x20);



        for (env = first_cpu; env != NULL; env = env->next_cpu) {

            if (env->cpu_index == i) {

                break;

            }

        }



        if (!env) {

            continue;

        }



        snprintf(cpu_name, sizeof(cpu_name), "/cpus/PowerPC,8544@%x", env->cpu_index);

        qemu_devtree_add_subnode(fdt, cpu_name);

        qemu_devtree_setprop_cell(fdt, cpu_name, "clock-frequency", clock_freq);

        qemu_devtree_setprop_cell(fdt, cpu_name, "timebase-frequency", tb_freq);

        qemu_devtree_setprop_string(fdt, cpu_name, "device_type", "cpu");

        qemu_devtree_setprop_cell(fdt, cpu_name, "reg", env->cpu_index);

        qemu_devtree_setprop_cell(fdt, cpu_name, "d-cache-line-size",

                                  env->dcache_line_size);

        qemu_devtree_setprop_cell(fdt, cpu_name, "i-cache-line-size",

                                  env->icache_line_size);

        qemu_devtree_setprop_cell(fdt, cpu_name, "d-cache-size", 0x8000);

        qemu_devtree_setprop_cell(fdt, cpu_name, "i-cache-size", 0x8000);

        qemu_devtree_setprop_cell(fdt, cpu_name, "bus-frequency", 0);

        if (env->cpu_index) {

            qemu_devtree_setprop_string(fdt, cpu_name, "status", "disabled");

            qemu_devtree_setprop_string(fdt, cpu_name, "enable-method", "spin-table");

            qemu_devtree_setprop_u64(fdt, cpu_name, "cpu-release-addr",

                                     cpu_release_addr);

        } else {

            qemu_devtree_setprop_string(fdt, cpu_name, "status", "okay");

        }

    }



    qemu_devtree_add_subnode(fdt, "/aliases");

    /* XXX These should go into their respective devices' code */

    snprintf(soc, sizeof(soc), "/soc8544@%x", MPC8544_CCSRBAR_BASE);

    qemu_devtree_add_subnode(fdt, soc);

    qemu_devtree_setprop_string(fdt, soc, "device_type", "soc");

    qemu_devtree_setprop_string(fdt, soc, "compatible", "simple-bus");

    qemu_devtree_setprop_cell(fdt, soc, "#address-cells", 1);

    qemu_devtree_setprop_cell(fdt, soc, "#size-cells", 1);

    qemu_devtree_setprop_cells(fdt, soc, "ranges", 0x0, MPC8544_CCSRBAR_BASE,

                               MPC8544_CCSRBAR_SIZE);

    qemu_devtree_setprop_cells(fdt, soc, "reg", MPC8544_CCSRBAR_BASE,

                               MPC8544_CCSRBAR_REGSIZE);

    /* XXX should contain a reasonable value */

    qemu_devtree_setprop_cell(fdt, soc, "bus-frequency", 0);



    snprintf(mpic, sizeof(mpic), "%s/pic@%x", soc,

             MPC8544_MPIC_REGS_BASE - MPC8544_CCSRBAR_BASE);

    qemu_devtree_add_subnode(fdt, mpic);

    qemu_devtree_setprop_string(fdt, mpic, "device_type", "open-pic");

    qemu_devtree_setprop_string(fdt, mpic, "compatible", "chrp,open-pic");

    qemu_devtree_setprop_cells(fdt, mpic, "reg", MPC8544_MPIC_REGS_BASE -

                               MPC8544_CCSRBAR_BASE, 0x40000);

    qemu_devtree_setprop_cell(fdt, mpic, "#address-cells", 0);

    qemu_devtree_setprop_cell(fdt, mpic, "#interrupt-cells", 2);

    mpic_ph = qemu_devtree_alloc_phandle(fdt);

    qemu_devtree_setprop_cell(fdt, mpic, "phandle", mpic_ph);

    qemu_devtree_setprop_cell(fdt, mpic, "linux,phandle", mpic_ph);

    qemu_devtree_setprop(fdt, mpic, "interrupt-controller", NULL, 0);



    /*

     * We have to generate ser1 first, because Linux takes the first

     * device it finds in the dt as serial output device. And we generate

     * devices in reverse order to the dt.

     */

    snprintf(ser1, sizeof(ser1), "%s/serial@%x", soc,

             MPC8544_SERIAL1_REGS_BASE - MPC8544_CCSRBAR_BASE);

    qemu_devtree_add_subnode(fdt, ser1);

    qemu_devtree_setprop_string(fdt, ser1, "device_type", "serial");

    qemu_devtree_setprop_string(fdt, ser1, "compatible", "ns16550");

    qemu_devtree_setprop_cells(fdt, ser1, "reg", MPC8544_SERIAL1_REGS_BASE -

                               MPC8544_CCSRBAR_BASE, 0x100);

    qemu_devtree_setprop_cell(fdt, ser1, "cell-index", 1);

    qemu_devtree_setprop_cell(fdt, ser1, "clock-frequency", 0);

    qemu_devtree_setprop_cells(fdt, ser1, "interrupts", 42, 2);

    qemu_devtree_setprop_phandle(fdt, ser1, "interrupt-parent", mpic);

    qemu_devtree_setprop_string(fdt, "/aliases", "serial1", ser1);



    snprintf(ser0, sizeof(ser0), "%s/serial@%x", soc,

             MPC8544_SERIAL0_REGS_BASE - MPC8544_CCSRBAR_BASE);

    qemu_devtree_add_subnode(fdt, ser0);

    qemu_devtree_setprop_string(fdt, ser0, "device_type", "serial");

    qemu_devtree_setprop_string(fdt, ser0, "compatible", "ns16550");

    qemu_devtree_setprop_cells(fdt, ser0, "reg", MPC8544_SERIAL0_REGS_BASE -

                               MPC8544_CCSRBAR_BASE, 0x100);

    qemu_devtree_setprop_cell(fdt, ser0, "cell-index", 0);

    qemu_devtree_setprop_cell(fdt, ser0, "clock-frequency", 0);

    qemu_devtree_setprop_cells(fdt, ser0, "interrupts", 42, 2);

    qemu_devtree_setprop_phandle(fdt, ser0, "interrupt-parent", mpic);

    qemu_devtree_setprop_string(fdt, "/aliases", "serial0", ser0);

    qemu_devtree_setprop_string(fdt, "/chosen", "linux,stdout-path", ser0);



    snprintf(gutil, sizeof(gutil), "%s/global-utilities@%x", soc,

             MPC8544_UTIL_BASE - MPC8544_CCSRBAR_BASE);

    qemu_devtree_add_subnode(fdt, gutil);

    qemu_devtree_setprop_string(fdt, gutil, "compatible", "fsl,mpc8544-guts");

    qemu_devtree_setprop_cells(fdt, gutil, "reg", MPC8544_UTIL_BASE -

                               MPC8544_CCSRBAR_BASE, 0x1000);

    qemu_devtree_setprop(fdt, gutil, "fsl,has-rstcr", NULL, 0);



    snprintf(pci, sizeof(pci), "/pci@%x", MPC8544_PCI_REGS_BASE);

    qemu_devtree_add_subnode(fdt, pci);

    qemu_devtree_setprop_cell(fdt, pci, "cell-index", 0);

    qemu_devtree_setprop_string(fdt, pci, "compatible", "fsl,mpc8540-pci");

    qemu_devtree_setprop_string(fdt, pci, "device_type", "pci");

    qemu_devtree_setprop_cells(fdt, pci, "interrupt-map-mask", 0xf800, 0x0,

                               0x0, 0x7);

    pci_map_create(fdt, pci_map, qemu_devtree_get_phandle(fdt, mpic));

    qemu_devtree_setprop(fdt, pci, "interrupt-map", pci_map, sizeof(pci_map));

    qemu_devtree_setprop_phandle(fdt, pci, "interrupt-parent", mpic);

    qemu_devtree_setprop_cells(fdt, pci, "interrupts", 24, 2);

    qemu_devtree_setprop_cells(fdt, pci, "bus-range", 0, 255);

    for (i = 0; i < 12; i++) {

        pci_ranges[i] = cpu_to_be32(pci_ranges[i]);

    }

    qemu_devtree_setprop(fdt, pci, "ranges", pci_ranges, sizeof(pci_ranges));

    qemu_devtree_setprop_cells(fdt, pci, "reg", MPC8544_PCI_REGS_BASE,

                               0x1000);

    qemu_devtree_setprop_cell(fdt, pci, "clock-frequency", 66666666);

    qemu_devtree_setprop_cell(fdt, pci, "#interrupt-cells", 1);

    qemu_devtree_setprop_cell(fdt, pci, "#size-cells", 2);

    qemu_devtree_setprop_cell(fdt, pci, "#address-cells", 3);

    qemu_devtree_setprop_string(fdt, "/aliases", "pci0", pci);



    machine_opts = qemu_opts_find(qemu_find_opts("machine"), 0);

    if (machine_opts) {

        dumpdtb = qemu_opt_get(machine_opts, "dumpdtb");

    }

    if (dumpdtb) {

        /* Dump the dtb to a file and quit */

        FILE *f = fopen(dumpdtb, "wb");

        size_t len;

        len = fwrite(fdt, fdt_size, 1, f);

        fclose(f);

        if (len != fdt_size) {

            exit(1);

        }

        exit(0);

    }



    ret = rom_add_blob_fixed(BINARY_DEVICE_TREE_FILE, fdt, fdt_size, addr);

    if (ret < 0) {

        goto out;

    }

    g_free(fdt);

    ret = fdt_size;



out:



    return ret;

}
