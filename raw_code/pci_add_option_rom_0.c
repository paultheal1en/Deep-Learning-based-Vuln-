static int pci_add_option_rom(PCIDevice *pdev, bool is_default_rom)

{

    int size;

    char *path;

    void *ptr;

    char name[32];

    const VMStateDescription *vmsd;



    if (!pdev->romfile)

        return 0;

    if (strlen(pdev->romfile) == 0)

        return 0;



    if (!pdev->rom_bar) {

        /*

         * Load rom via fw_cfg instead of creating a rom bar,

         * for 0.11 compatibility.

         */

        int class = pci_get_word(pdev->config + PCI_CLASS_DEVICE);

        if (class == 0x0300) {

            rom_add_vga(pdev->romfile);

        } else {

            rom_add_option(pdev->romfile, -1);

        }

        return 0;

    }



    path = qemu_find_file(QEMU_FILE_TYPE_BIOS, pdev->romfile);

    if (path == NULL) {

        path = g_strdup(pdev->romfile);

    }



    size = get_image_size(path);

    if (size < 0) {

        error_report("%s: failed to find romfile \"%s\"",

                     __FUNCTION__, pdev->romfile);

        g_free(path);

        return -1;

    }

    if (size & (size - 1)) {

        size = 1 << qemu_fls(size);

    }



    vmsd = qdev_get_vmsd(DEVICE(pdev));



    if (vmsd) {

        snprintf(name, sizeof(name), "%s.rom", vmsd->name);

    } else {

        snprintf(name, sizeof(name), "%s.rom", object_get_typename(OBJECT(pdev)));

    }

    pdev->has_rom = true;

    memory_region_init_ram(&pdev->rom, name, size);

    vmstate_register_ram(&pdev->rom, &pdev->qdev);

    ptr = memory_region_get_ram_ptr(&pdev->rom);

    load_image(path, ptr);

    g_free(path);



    if (is_default_rom) {

        /* Only the default rom images will be patched (if needed). */

        pci_patch_ids(pdev, ptr, size);

    }



    qemu_put_ram_ptr(ptr);



    pci_register_bar(pdev, PCI_ROM_SLOT, 0, &pdev->rom);



    return 0;

}
