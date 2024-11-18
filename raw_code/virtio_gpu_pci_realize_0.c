static void virtio_gpu_pci_realize(VirtIOPCIProxy *vpci_dev, Error **errp)

{

    VirtIOGPUPCI *vgpu = VIRTIO_GPU_PCI(vpci_dev);

    VirtIOGPU *g = &vgpu->vdev;

    DeviceState *vdev = DEVICE(&vgpu->vdev);

    int i;



    qdev_set_parent_bus(vdev, BUS(&vpci_dev->bus));

    /* force virtio-1.0 */

    vpci_dev->flags &= ~VIRTIO_PCI_FLAG_DISABLE_MODERN;

    vpci_dev->flags |= VIRTIO_PCI_FLAG_DISABLE_LEGACY;

    object_property_set_bool(OBJECT(vdev), true, "realized", errp);



    for (i = 0; i < g->conf.max_outputs; i++) {

        object_property_set_link(OBJECT(g->scanout[i].con),

                                 OBJECT(vpci_dev),

                                 "device", errp);

    }

}
