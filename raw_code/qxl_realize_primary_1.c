static void qxl_realize_primary(PCIDevice *dev, Error **errp)

{

    PCIQXLDevice *qxl = PCI_QXL(dev);

    VGACommonState *vga = &qxl->vga;

    Error *local_err = NULL;



    qxl->id = 0;

    qxl_init_ramsize(qxl);

    vga->vbe_size = qxl->vgamem_size;

    vga->vram_size_mb = qxl->vga.vram_size >> 20;

    vga_common_init(vga, OBJECT(dev), true);

    vga_init(vga, OBJECT(dev),

             pci_address_space(dev), pci_address_space_io(dev), false);

    portio_list_init(&qxl->vga_port_list, OBJECT(dev), qxl_vga_portio_list,

                     vga, "vga");

    portio_list_set_flush_coalesced(&qxl->vga_port_list);

    portio_list_add(&qxl->vga_port_list, pci_address_space_io(dev), 0x3b0);



    vga->con = graphic_console_init(DEVICE(dev), 0, &qxl_ops, qxl);

    qemu_spice_display_init_common(&qxl->ssd);



    qxl_realize_common(qxl, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    qxl->ssd.dcl.ops = &display_listener_ops;

    qxl->ssd.dcl.con = vga->con;

    register_displaychangelistener(&qxl->ssd.dcl);

}
