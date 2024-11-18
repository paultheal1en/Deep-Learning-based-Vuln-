static void s390_virtio_device_sync(VirtIOS390Device *dev)

{

    VirtIOS390Bus *bus = DO_UPCAST(VirtIOS390Bus, bus, dev->qdev.parent_bus);

    ram_addr_t cur_offs;

    uint8_t num_vq;

    int i;



    virtio_reset(dev->vdev);



    /* Sync dev space */

    stb_phys(dev->dev_offs + VIRTIO_DEV_OFFS_TYPE, dev->vdev->device_id);



    stb_phys(dev->dev_offs + VIRTIO_DEV_OFFS_NUM_VQ, s390_virtio_device_num_vq(dev));

    stb_phys(dev->dev_offs + VIRTIO_DEV_OFFS_FEATURE_LEN, dev->feat_len);



    stb_phys(dev->dev_offs + VIRTIO_DEV_OFFS_CONFIG_LEN, dev->vdev->config_len);



    num_vq = s390_virtio_device_num_vq(dev);

    stb_phys(dev->dev_offs + VIRTIO_DEV_OFFS_NUM_VQ, num_vq);



    /* Sync virtqueues */

    for (i = 0; i < num_vq; i++) {

        ram_addr_t vq = (dev->dev_offs + VIRTIO_DEV_OFFS_CONFIG) +

                        (i * VIRTIO_VQCONFIG_LEN);

        ram_addr_t vring;



        vring = s390_virtio_next_ring(bus);

        virtio_queue_set_addr(dev->vdev, i, vring);

        virtio_queue_set_vector(dev->vdev, i, i);

        stq_phys(vq + VIRTIO_VQCONFIG_OFFS_ADDRESS, vring);

        stw_phys(vq + VIRTIO_VQCONFIG_OFFS_NUM, virtio_queue_get_num(dev->vdev, i));

    }



    cur_offs = dev->dev_offs;

    cur_offs += VIRTIO_DEV_OFFS_CONFIG;

    cur_offs += num_vq * VIRTIO_VQCONFIG_LEN;



    /* Sync feature bitmap */

    if (dev->vdev->get_features) {

        stl_phys(cur_offs, dev->vdev->get_features(dev->vdev));

    }



    dev->feat_offs = cur_offs + dev->feat_len;

    cur_offs += dev->feat_len * 2;



    /* Sync config space */

    if (dev->vdev->get_config) {

        dev->vdev->get_config(dev->vdev, dev->vdev->config);

    }



    cpu_physical_memory_rw(cur_offs, dev->vdev->config, dev->vdev->config_len, 1);

    cur_offs += dev->vdev->config_len;

}
