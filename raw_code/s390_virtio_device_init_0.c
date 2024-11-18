static int s390_virtio_device_init(VirtIOS390Device *dev, VirtIODevice *vdev)

{

    VirtIOS390Bus *bus;

    int dev_len;



    bus = DO_UPCAST(VirtIOS390Bus, bus, dev->qdev.parent_bus);

    dev->vdev = vdev;

    dev->dev_offs = bus->dev_offs;

    dev->feat_len = sizeof(uint32_t); /* always keep 32 bits features */



    dev_len = VIRTIO_DEV_OFFS_CONFIG;

    dev_len += s390_virtio_device_num_vq(dev) * VIRTIO_VQCONFIG_LEN;

    dev_len += dev->feat_len * 2;

    dev_len += virtio_bus_get_vdev_config_len(&dev->bus);



    bus->dev_offs += dev_len;



    dev->host_features = virtio_bus_get_vdev_features(&dev->bus,

                                                      dev->host_features);

    s390_virtio_device_sync(dev);

    s390_virtio_reset_idx(dev);

    if (dev->qdev.hotplugged) {

        S390CPU *cpu = s390_cpu_addr2state(0);

        s390_virtio_irq(cpu, VIRTIO_PARAM_DEV_ADD, dev->dev_offs);

    }



    return 0;

}
