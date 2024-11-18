static void qxl_reset_surfaces(PCIQXLDevice *d)

{

    dprint(d, 1, "%s:\n", __FUNCTION__);

    d->mode = QXL_MODE_UNDEFINED;

    qemu_mutex_unlock_iothread();

    d->ssd.worker->destroy_surfaces(d->ssd.worker);

    qemu_mutex_lock_iothread();

    memset(&d->guest_surfaces.cmds, 0, sizeof(d->guest_surfaces.cmds));

}
