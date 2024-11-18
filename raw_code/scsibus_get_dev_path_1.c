static char *scsibus_get_dev_path(DeviceState *dev)

{

    SCSIDevice *d = DO_UPCAST(SCSIDevice, qdev, dev);

    DeviceState *hba = dev->parent_bus->parent;

    char *id = NULL;



    if (hba && hba->parent_bus && hba->parent_bus->info->get_dev_path) {

        id = hba->parent_bus->info->get_dev_path(hba);

    }

    if (id) {

        return g_strdup_printf("%s/%d:%d:%d", id, d->channel, d->id, d->lun);

    } else {

        return g_strdup_printf("%d:%d:%d", d->channel, d->id, d->lun);

    }

}
