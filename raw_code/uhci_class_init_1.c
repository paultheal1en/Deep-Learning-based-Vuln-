static void uhci_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    UHCIPCIDeviceClass *u = container_of(k, UHCIPCIDeviceClass, parent_class);

    UHCIInfo *info = data;



    k->init = info->initfn ? info->initfn : usb_uhci_common_initfn;

    k->exit = info->unplug ? usb_uhci_exit : NULL;

    k->vendor_id = info->vendor_id;

    k->device_id = info->device_id;

    k->revision  = info->revision;

    k->class_id  = PCI_CLASS_SERIAL_USB;


    dc->vmsd = &vmstate_uhci;

    dc->props = uhci_properties;

    u->info = *info;

}