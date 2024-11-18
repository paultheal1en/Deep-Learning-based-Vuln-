static void virtio_blk_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    dc->exit = virtio_blk_device_exit;

    dc->props = virtio_blk_properties;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);

    vdc->init = virtio_blk_device_init;

    vdc->get_config = virtio_blk_update_config;

    vdc->set_config = virtio_blk_set_config;

    vdc->get_features = virtio_blk_get_features;

    vdc->set_status = virtio_blk_set_status;

    vdc->reset = virtio_blk_reset;

}
