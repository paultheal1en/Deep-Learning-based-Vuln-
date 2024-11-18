static void lsi_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = lsi_scsi_init;

    k->exit = lsi_scsi_uninit;

    k->vendor_id = PCI_VENDOR_ID_LSI_LOGIC;

    k->device_id = PCI_DEVICE_ID_LSI_53C895A;

    k->class_id = PCI_CLASS_STORAGE_SCSI;

    k->subsystem_id = 0x1000;

    dc->alias = "lsi";

    dc->reset = lsi_scsi_reset;

    dc->vmsd = &vmstate_lsi_scsi;

}
