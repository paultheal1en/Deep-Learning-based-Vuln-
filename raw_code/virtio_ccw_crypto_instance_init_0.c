static void virtio_ccw_crypto_instance_init(Object *obj)

{

    VirtIOCryptoCcw *dev = VIRTIO_CRYPTO_CCW(obj);

    VirtioCcwDevice *ccw_dev = VIRTIO_CCW_DEVICE(obj);



    ccw_dev->force_revision_1 = true;

    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),

                                TYPE_VIRTIO_CRYPTO);



    object_property_add_alias(obj, "cryptodev", OBJECT(&dev->vdev),

                              "cryptodev", &error_abort);

}
