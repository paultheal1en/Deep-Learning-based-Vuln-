static void scsi_generic_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SCSIDeviceClass *sc = SCSI_DEVICE_CLASS(klass);



    sc->realize      = scsi_generic_realize;

    sc->unrealize    = scsi_unrealize;

    sc->alloc_req    = scsi_new_request;

    sc->parse_cdb    = scsi_generic_parse_cdb;

    dc->fw_name = "disk";

    dc->desc = "pass through generic scsi device (/dev/sg*)";

    dc->reset = scsi_generic_reset;

    dc->props = scsi_generic_properties;

    dc->vmsd  = &vmstate_scsi_device;

}
