static void create_flash(const VirtBoardInfo *vbi)

{

    /* Create two flash devices to fill the VIRT_FLASH space in the memmap.

     * Any file passed via -bios goes in the first of these.

     */

    hwaddr flashsize = vbi->memmap[VIRT_FLASH].size / 2;

    hwaddr flashbase = vbi->memmap[VIRT_FLASH].base;

    char *nodename;



    if (bios_name) {

        char *fn;

        int image_size;



        if (drive_get(IF_PFLASH, 0, 0)) {

            error_report("The contents of the first flash device may be "

                         "specified with -bios or with -drive if=pflash... "

                         "but you cannot use both options at once");

            exit(1);

        }

        fn = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        if (!fn) {

            error_report("Could not find ROM image '%s'", bios_name);

            exit(1);

        }

        image_size = load_image_targphys(fn, flashbase, flashsize);

        g_free(fn);

        if (image_size < 0) {

            error_report("Could not load ROM image '%s'", bios_name);

            exit(1);

        }

        g_free(fn);

    }



    create_one_flash("virt.flash0", flashbase, flashsize);

    create_one_flash("virt.flash1", flashbase + flashsize, flashsize);



    nodename = g_strdup_printf("/flash@%" PRIx64, flashbase);

    qemu_fdt_add_subnode(vbi->fdt, nodename);

    qemu_fdt_setprop_string(vbi->fdt, nodename, "compatible", "cfi-flash");

    qemu_fdt_setprop_sized_cells(vbi->fdt, nodename, "reg",

                                 2, flashbase, 2, flashsize,

                                 2, flashbase + flashsize, 2, flashsize);

    qemu_fdt_setprop_cell(vbi->fdt, nodename, "bank-width", 4);

    g_free(nodename);

}
