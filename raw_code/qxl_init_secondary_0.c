static int qxl_init_secondary(PCIDevice *dev)

{

    static int device_id = 1;

    PCIQXLDevice *qxl = DO_UPCAST(PCIQXLDevice, pci, dev);

    ram_addr_t ram_size = msb_mask(qxl->vga.vram_size * 2 - 1);



    qxl->id = device_id++;



    if (ram_size < 16 * 1024 * 1024) {

        ram_size = 16 * 1024 * 1024;

    }

    qxl->vga.vram_size = ram_size;

    qxl->vga.vram_offset = qemu_ram_alloc(&qxl->pci.qdev, "qxl.vgavram",

                                          qxl->vga.vram_size);

    qxl->vga.vram_ptr = qemu_get_ram_ptr(qxl->vga.vram_offset);



    pci_config_set_class(dev->config, PCI_CLASS_DISPLAY_OTHER);

    return qxl_init_common(qxl);

}
