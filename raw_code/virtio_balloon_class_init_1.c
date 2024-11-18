static void virtio_balloon_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    dc->exit = virtio_balloon_device_exit;

    dc->props = virtio_balloon_properties;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    vdc->init = virtio_balloon_device_init;

    vdc->get_config = virtio_balloon_get_config;

    vdc->set_config = virtio_balloon_set_config;

    vdc->get_features = virtio_balloon_get_features;

}
