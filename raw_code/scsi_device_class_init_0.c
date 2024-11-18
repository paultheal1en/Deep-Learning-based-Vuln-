static void scsi_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *k = DEVICE_CLASS(klass);

    set_bit(DEVICE_CATEGORY_STORAGE, k->categories);

    k->bus_type = TYPE_SCSI_BUS;

    k->init     = scsi_qdev_init;

    k->unplug   = scsi_qdev_unplug;

    k->exit     = scsi_qdev_exit;

    k->props    = scsi_props;

}
