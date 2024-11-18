static void qdev_reset(void *opaque)

{

    DeviceState *dev = opaque;

    if (dev->info->reset)

        dev->info->reset(dev);

}
