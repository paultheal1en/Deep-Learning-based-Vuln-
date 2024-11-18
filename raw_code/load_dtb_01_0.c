static int load_dtb(target_phys_addr_t addr, const struct arm_boot_info *binfo)

{

#ifdef CONFIG_FDT

    uint32_t *mem_reg_property;

    uint32_t mem_reg_propsize;

    void *fdt = NULL;

    char *filename;

    int size, rc;

    uint32_t acells, scells, hival;



    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, binfo->dtb_filename);

    if (!filename) {

        fprintf(stderr, "Couldn't open dtb file %s\n", binfo->dtb_filename);

        return -1;

    }



    fdt = load_device_tree(filename, &size);

    if (!fdt) {

        fprintf(stderr, "Couldn't open dtb file %s\n", filename);

        g_free(filename);

        return -1;

    }

    g_free(filename);



    acells = qemu_devtree_getprop_cell(fdt, "/", "#address-cells");

    scells = qemu_devtree_getprop_cell(fdt, "/", "#size-cells");

    if (acells == 0 || scells == 0) {

        fprintf(stderr, "dtb file invalid (#address-cells or #size-cells 0)\n");

        return -1;

    }



    mem_reg_propsize = acells + scells;

    mem_reg_property = g_new0(uint32_t, mem_reg_propsize);

    mem_reg_property[acells - 1] = cpu_to_be32(binfo->loader_start);

    hival = cpu_to_be32(binfo->loader_start >> 32);

    if (acells > 1) {

        mem_reg_property[acells - 2] = hival;

    } else if (hival != 0) {

        fprintf(stderr, "qemu: dtb file not compatible with "

                "RAM start address > 4GB\n");

        exit(1);

    }

    mem_reg_property[acells + scells - 1] = cpu_to_be32(binfo->ram_size);

    hival = cpu_to_be32(binfo->ram_size >> 32);

    if (scells > 1) {

        mem_reg_property[acells + scells - 2] = hival;

    } else if (hival != 0) {

        fprintf(stderr, "qemu: dtb file not compatible with "

                "RAM size > 4GB\n");

        exit(1);

    }



    rc = qemu_devtree_setprop(fdt, "/memory", "reg", mem_reg_property,

                              mem_reg_propsize * sizeof(uint32_t));

    if (rc < 0) {

        fprintf(stderr, "couldn't set /memory/reg\n");

    }



    if (binfo->kernel_cmdline && *binfo->kernel_cmdline) {

        rc = qemu_devtree_setprop_string(fdt, "/chosen", "bootargs",

                                          binfo->kernel_cmdline);

        if (rc < 0) {

            fprintf(stderr, "couldn't set /chosen/bootargs\n");

        }

    }



    if (binfo->initrd_size) {

        rc = qemu_devtree_setprop_cell(fdt, "/chosen", "linux,initrd-start",

                binfo->loader_start + INITRD_LOAD_ADDR);

        if (rc < 0) {

            fprintf(stderr, "couldn't set /chosen/linux,initrd-start\n");

        }



        rc = qemu_devtree_setprop_cell(fdt, "/chosen", "linux,initrd-end",

                    binfo->loader_start + INITRD_LOAD_ADDR +

                    binfo->initrd_size);

        if (rc < 0) {

            fprintf(stderr, "couldn't set /chosen/linux,initrd-end\n");

        }

    }



    cpu_physical_memory_write(addr, fdt, size);



    return 0;



#else

    fprintf(stderr, "Device tree requested, "

                "but qemu was compiled without fdt support\n");

    return -1;

#endif

}
