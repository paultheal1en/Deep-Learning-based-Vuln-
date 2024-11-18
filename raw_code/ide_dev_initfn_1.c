static int ide_dev_initfn(IDEDevice *dev, IDEDriveKind kind)
{
    IDEBus *bus = DO_UPCAST(IDEBus, qbus, dev->qdev.parent_bus);
    IDEState *s = bus->ifs + dev->unit;
    Error *err = NULL;
    if (dev->conf.discard_granularity == -1) {
        dev->conf.discard_granularity = 512;
    } else if (dev->conf.discard_granularity &&
               dev->conf.discard_granularity != 512) {
        error_report("discard_granularity must be 512 for ide");
    blkconf_serial(&dev->conf, &dev->serial);
    if (kind != IDE_CD) {
        blkconf_geometry(&dev->conf, &dev->chs_trans, 65536, 16, 255, &err);
        if (err) {
            error_report("%s", error_get_pretty(err));
            error_free(err);
    if (ide_init_drive(s, dev->conf.blk, kind,
                       dev->version, dev->serial, dev->model, dev->wwn,
                       dev->conf.cyls, dev->conf.heads, dev->conf.secs,
                       dev->chs_trans) < 0) {
    if (!dev->version) {
        dev->version = g_strdup(s->version);
    if (!dev->serial) {
        dev->serial = g_strdup(s->drive_serial_str);
    add_boot_device_path(dev->conf.bootindex, &dev->qdev,
                         dev->unit ? "/disk@1" : "/disk@0");
    return 0;