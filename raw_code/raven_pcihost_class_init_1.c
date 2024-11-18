static void raven_pcihost_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);

    dc->realize = raven_pcihost_realizefn;

    dc->fw_name = "pci";

    dc->no_user = 1;

}
