IDEDevice *ide_create_drive(IDEBus *bus, int unit, DriveInfo *drive)

{

    DeviceState *dev;



    dev = qdev_create(&bus->qbus, drive->media_cd ? "ide-cd" : "ide-hd");

    qdev_prop_set_uint32(dev, "unit", unit);

    qdev_prop_set_drive_nofail(dev, "drive",

                               blk_bs(blk_by_legacy_dinfo(drive)));

    qdev_init_nofail(dev);

    return DO_UPCAST(IDEDevice, qdev, dev);

}
