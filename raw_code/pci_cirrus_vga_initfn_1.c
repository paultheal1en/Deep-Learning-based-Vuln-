static int pci_cirrus_vga_initfn(PCIDevice *dev)

{

     PCICirrusVGAState *d = DO_UPCAST(PCICirrusVGAState, dev, dev);

     CirrusVGAState *s = &d->cirrus_vga;

     PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(dev);

     int16_t device_id = pc->device_id;











     /* setup VGA */

     vga_common_init(&s->vga, OBJECT(dev), true);

     cirrus_init_common(s, OBJECT(dev), device_id, 1, pci_address_space(dev),

                        pci_address_space_io(dev));

     s->vga.con = graphic_console_init(DEVICE(dev), 0, s->vga.hw_ops, &s->vga);



     /* setup PCI */



    memory_region_init(&s->pci_bar, OBJECT(dev), "cirrus-pci-bar0", 0x2000000);



    /* XXX: add byte swapping apertures */

    memory_region_add_subregion(&s->pci_bar, 0, &s->cirrus_linear_io);

    memory_region_add_subregion(&s->pci_bar, 0x1000000,

                                &s->cirrus_linear_bitblt_io);



     /* setup memory space */

     /* memory #0 LFB */

     /* memory #1 memory-mapped I/O */

     /* XXX: s->vga.vram_size must be a power of two */

     pci_register_bar(&d->dev, 0, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->pci_bar);

     if (device_id == CIRRUS_ID_CLGD5446) {

         pci_register_bar(&d->dev, 1, 0, &s->cirrus_mmio_io);


     return 0;
