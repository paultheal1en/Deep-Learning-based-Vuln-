static void virtio_vga_realize(VirtIOPCIProxy *vpci_dev, Error **errp)

{

    VirtIOVGA *vvga = VIRTIO_VGA(vpci_dev);

    VirtIOGPU *g = &vvga->vdev;

    VGACommonState *vga = &vvga->vga;

    Error *err = NULL;

    uint32_t offset;

    int i;



    /* init vga compat bits */

    vga->vram_size_mb = 8;

    vga_common_init(vga, OBJECT(vpci_dev), false);

    vga_init(vga, OBJECT(vpci_dev), pci_address_space(&vpci_dev->pci_dev),

             pci_address_space_io(&vpci_dev->pci_dev), true);

    pci_register_bar(&vpci_dev->pci_dev, 0,

                     PCI_BASE_ADDRESS_MEM_PREFETCH, &vga->vram);



    /*

     * Configure virtio bar and regions

     *

     * We use bar #2 for the mmio regions, to be compatible with stdvga.

     * virtio regions are moved to the end of bar #2, to make room for

     * the stdvga mmio registers at the start of bar #2.

     */

    vpci_dev->modern_mem_bar = 2;

    vpci_dev->msix_bar = 4;

    offset = memory_region_size(&vpci_dev->modern_bar);

    offset -= vpci_dev->notify.size;

    vpci_dev->notify.offset = offset;

    offset -= vpci_dev->device.size;

    vpci_dev->device.offset = offset;

    offset -= vpci_dev->isr.size;

    vpci_dev->isr.offset = offset;

    offset -= vpci_dev->common.size;

    vpci_dev->common.offset = offset;



    /* init virtio bits */

    qdev_set_parent_bus(DEVICE(g), BUS(&vpci_dev->bus));

    /* force virtio-1.0 */

    vpci_dev->flags &= ~VIRTIO_PCI_FLAG_DISABLE_MODERN;

    vpci_dev->flags |= VIRTIO_PCI_FLAG_DISABLE_LEGACY;

    object_property_set_bool(OBJECT(g), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    /* add stdvga mmio regions */

    pci_std_vga_mmio_region_init(vga, &vpci_dev->modern_bar,

                                 vvga->vga_mrs, true);



    vga->con = g->scanout[0].con;

    graphic_console_set_hwops(vga->con, &virtio_vga_ops, vvga);



    for (i = 0; i < g->conf.max_outputs; i++) {

        object_property_set_link(OBJECT(g->scanout[i].con),

                                 OBJECT(vpci_dev),

                                 "device", errp);

    }

}
