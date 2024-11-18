static void scsi_block_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SCSIDeviceClass *sc = SCSI_DEVICE_CLASS(klass);



    sc->realize      = scsi_block_realize;

    sc->unrealize    = scsi_unrealize;

    sc->alloc_req    = scsi_block_new_request;

    sc->parse_cdb    = scsi_block_parse_cdb;

    dc->fw_name = "disk";

    dc->desc = "SCSI block device passthrough";

    dc->reset = scsi_disk_reset;

    dc->props = scsi_block_properties;

    dc->vmsd  = &vmstate_scsi_disk_state;

}
