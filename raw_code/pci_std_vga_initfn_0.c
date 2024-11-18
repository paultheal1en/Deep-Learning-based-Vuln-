static int pci_std_vga_initfn(PCIDevice *dev)

{

    PCIVGAState *d = DO_UPCAST(PCIVGAState, dev, dev);

    VGACommonState *s = &d->vga;



    /* vga + console init */

    vga_common_init(s);

    vga_init(s, pci_address_space(dev), pci_address_space_io(dev), true);



    s->con = graphic_console_init(s->update, s->invalidate,

                                  s->screen_dump, s->text_update, s);



    /* XXX: VGA_RAM_SIZE must be a power of two */

    pci_register_bar(&d->dev, 0, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vram);



    /* mmio bar for vga register access */

    if (d->flags & (1 << PCI_VGA_FLAG_ENABLE_MMIO)) {

        memory_region_init(&d->mmio, "vga.mmio", 4096);

        memory_region_init_io(&d->ioport, &pci_vga_ioport_ops, d,

                              "vga ioports remapped", PCI_VGA_IOPORT_SIZE);

        memory_region_init_io(&d->bochs, &pci_vga_bochs_ops, d,

                              "bochs dispi interface", PCI_VGA_BOCHS_SIZE);



        memory_region_add_subregion(&d->mmio, PCI_VGA_IOPORT_OFFSET,

                                    &d->ioport);

        memory_region_add_subregion(&d->mmio, PCI_VGA_BOCHS_OFFSET,

                                    &d->bochs);

        pci_register_bar(&d->dev, 2, PCI_BASE_ADDRESS_SPACE_MEMORY, &d->mmio);

    }



    if (!dev->rom_bar) {

        /* compatibility with pc-0.13 and older */

        vga_init_vbe(s, pci_address_space(dev));

    }



    return 0;

}
