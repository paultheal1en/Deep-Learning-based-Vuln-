int qdev_unplug(DeviceState *dev)
{
    if (!dev->parent_bus->allow_hotplug) {
        qemu_error("Bus %s does not support hotplugging\n",
                   dev->parent_bus->name);
        return -1;
    }
    return dev->info->unplug(dev);
}