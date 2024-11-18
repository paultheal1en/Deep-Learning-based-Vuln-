static uint32_t virtio_ioport_read(void *opaque, uint32_t addr)

{

    VirtIODevice *vdev = to_virtio_device(opaque);

    uint32_t ret = 0xFFFFFFFF;



    addr -= vdev->addr;



    switch (addr) {

    case VIRTIO_PCI_HOST_FEATURES:

        ret = vdev->get_features(vdev);

        ret |= (1 << VIRTIO_F_NOTIFY_ON_EMPTY);

        break;

    case VIRTIO_PCI_GUEST_FEATURES:

        ret = vdev->features;

        break;

    case VIRTIO_PCI_QUEUE_PFN:

        ret = vdev->vq[vdev->queue_sel].pfn;

        break;

    case VIRTIO_PCI_QUEUE_NUM:

        ret = vdev->vq[vdev->queue_sel].vring.num;

        break;

    case VIRTIO_PCI_QUEUE_SEL:

        ret = vdev->queue_sel;

        break;

    case VIRTIO_PCI_STATUS:

        ret = vdev->status;

        break;

    case VIRTIO_PCI_ISR:

        /* reading from the ISR also clears it. */

        ret = vdev->isr;

        vdev->isr = 0;

        virtio_update_irq(vdev);

        break;

    default:

        break;

    }



    return ret;

}
