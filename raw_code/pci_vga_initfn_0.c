static int pci_vga_initfn(PCIDevice *dev)

{

     PCIVGAState *d = DO_UPCAST(PCIVGAState, dev, dev);

     VGACommonState *s = &d->vga;

     uint8_t *pci_conf = d->dev.config;



     // vga + console init

     vga_common_init(s, VGA_RAM_SIZE);

     vga_init(s);



     s->ds = graphic_console_init(s->update, s->invalidate,

                                  s->screen_dump, s->text_update, s);



     // dummy VGA (same as Bochs ID)

     pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_QEMU);

     pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_QEMU_VGA);

     pci_config_set_class(pci_conf, PCI_CLASS_DISPLAY_VGA);



     /* XXX: VGA_RAM_SIZE must be a power of two */

     pci_register_bar(&d->dev, 0, VGA_RAM_SIZE,

                      PCI_BASE_ADDRESS_MEM_PREFETCH, vga_map);



     if (!dev->rom_bar) {

         /* compatibility with pc-0.13 and older */

         vga_init_vbe(s);

     }



     return 0;

}
