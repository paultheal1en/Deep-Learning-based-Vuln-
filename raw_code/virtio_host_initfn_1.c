static void virtio_host_initfn(Object *obj)

{

    VirtIOInputHostPCI *dev = VIRTIO_INPUT_HOST_PCI(obj);



    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),

                                TYPE_VIRTIO_INPUT_HOST);

}
