void qdev_free(DeviceState *dev)

{

#if 0 /* FIXME: need sane vmstate_unregister function */

    if (dev->info->vmsd)

        vmstate_unregister(dev->info->vmsd, dev);

#endif

    if (dev->info->reset)

        qemu_unregister_reset(dev->info->reset, dev);

    LIST_REMOVE(dev, sibling);

    qemu_free(dev);

}
