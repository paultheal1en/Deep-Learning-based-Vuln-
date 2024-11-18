static void bonito_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = bonito_initfn;

    k->vendor_id = 0xdf53;

    k->device_id = 0x00d5;

    k->revision = 0x01;

    k->class_id = PCI_CLASS_BRIDGE_HOST;

    dc->desc = "Host bridge";

    dc->no_user = 1;

    dc->vmsd = &vmstate_bonito;

}
