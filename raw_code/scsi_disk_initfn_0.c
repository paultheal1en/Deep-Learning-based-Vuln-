static int scsi_disk_initfn(SCSIDevice *dev)

{

    SCSIDriveKind kind;



    if (!dev->conf.bs) {

        kind = SCSI_HD;         /* will die in scsi_initfn() */

    } else {

        kind = bdrv_get_type_hint(dev->conf.bs) == BDRV_TYPE_CDROM

            ? SCSI_CD : SCSI_HD;

    }



    return scsi_initfn(dev, kind);

}
