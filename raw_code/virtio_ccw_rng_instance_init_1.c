static void virtio_ccw_rng_instance_init(Object *obj)

{

    VirtIORNGCcw *dev = VIRTIO_RNG_CCW(obj);

    object_initialize(&dev->vdev, sizeof(dev->vdev), TYPE_VIRTIO_RNG);

    object_property_add_child(obj, "virtio-backend", OBJECT(&dev->vdev), NULL);

    object_property_add_link(obj, "rng", TYPE_RNG_BACKEND,

                             (Object **)&dev->vdev.conf.rng, NULL);

}
