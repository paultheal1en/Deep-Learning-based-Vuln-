static char *scsibus_get_fw_dev_path(DeviceState *dev)

{

    SCSIDevice *d = SCSI_DEVICE(dev);

    char path[100];



    snprintf(path, sizeof(path), "channel@%x/%s@%x,%x", d->channel,

             qdev_fw_name(dev), d->id, d->lun);



    return strdup(path);

}
