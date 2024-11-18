static void xen_platform_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = xen_platform_initfn;

    k->vendor_id = PCI_VENDOR_ID_XEN;

    k->device_id = PCI_DEVICE_ID_XEN_PLATFORM;

    k->class_id = PCI_CLASS_OTHERS << 8 | 0x80;

    k->subsystem_vendor_id = PCI_VENDOR_ID_XEN;

    k->subsystem_id = PCI_DEVICE_ID_XEN_PLATFORM;

    k->revision = 1;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    dc->desc = "XEN platform pci device";

    dc->reset = platform_reset;

    dc->vmsd = &vmstate_xen_platform;

}
