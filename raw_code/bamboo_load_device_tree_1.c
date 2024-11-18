static int bamboo_load_device_tree(hwaddr addr,

                                     uint32_t ramsize,

                                     hwaddr initrd_base,

                                     hwaddr initrd_size,

                                     const char *kernel_cmdline)

{

    int ret = -1;

    uint32_t mem_reg_property[] = { 0, 0, cpu_to_be32(ramsize) };

    char *filename;

    int fdt_size;

    void *fdt;

    uint32_t tb_freq = 400000000;

    uint32_t clock_freq = 400000000;



    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, BINARY_DEVICE_TREE_FILE);

    if (!filename) {

        goto out;

    }

    fdt = load_device_tree(filename, &fdt_size);

    g_free(filename);

    if (fdt == NULL) {

        goto out;

    }



    /* Manipulate device tree in memory. */



    ret = qemu_devtree_setprop(fdt, "/memory", "reg", mem_reg_property,

                               sizeof(mem_reg_property));

    if (ret < 0)

        fprintf(stderr, "couldn't set /memory/reg\n");



    ret = qemu_devtree_setprop_cell(fdt, "/chosen", "linux,initrd-start",

                                    initrd_base);

    if (ret < 0)

        fprintf(stderr, "couldn't set /chosen/linux,initrd-start\n");



    ret = qemu_devtree_setprop_cell(fdt, "/chosen", "linux,initrd-end",

                                    (initrd_base + initrd_size));

    if (ret < 0)

        fprintf(stderr, "couldn't set /chosen/linux,initrd-end\n");



    ret = qemu_devtree_setprop_string(fdt, "/chosen", "bootargs",

                                      kernel_cmdline);

    if (ret < 0)

        fprintf(stderr, "couldn't set /chosen/bootargs\n");



    /* Copy data from the host device tree into the guest. Since the guest can

     * directly access the timebase without host involvement, we must expose

     * the correct frequencies. */

    if (kvm_enabled()) {

        tb_freq = kvmppc_get_tbfreq();

        clock_freq = kvmppc_get_clockfreq();

    }



    qemu_devtree_setprop_cell(fdt, "/cpus/cpu@0", "clock-frequency",

                              clock_freq);

    qemu_devtree_setprop_cell(fdt, "/cpus/cpu@0", "timebase-frequency",

                              tb_freq);



    ret = rom_add_blob_fixed(BINARY_DEVICE_TREE_FILE, fdt, fdt_size, addr);

    g_free(fdt);



out:



    return ret;

}
