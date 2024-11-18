static uint32_t syborg_virtio_readl(void *opaque, target_phys_addr_t offset)

{

    SyborgVirtIOProxy *s = opaque;

    VirtIODevice *vdev = s->vdev;

    uint32_t ret;



    DPRINTF("readl 0x%x\n", (int)offset);

    if (offset >= SYBORG_VIRTIO_CONFIG) {

        return virtio_config_readl(vdev, offset - SYBORG_VIRTIO_CONFIG);

    }

    switch(offset >> 2) {

    case SYBORG_VIRTIO_ID:

        ret = SYBORG_ID_VIRTIO;

        break;

    case SYBORG_VIRTIO_DEVTYPE:

        ret = s->id;

        break;

    case SYBORG_VIRTIO_HOST_FEATURES:

        ret = vdev->get_features(vdev);

        ret |= (1 << VIRTIO_F_NOTIFY_ON_EMPTY);

        break;

    case SYBORG_VIRTIO_GUEST_FEATURES:

        ret = vdev->features;

        break;

    case SYBORG_VIRTIO_QUEUE_BASE:

        ret = virtio_queue_get_addr(vdev, vdev->queue_sel);

        break;

    case SYBORG_VIRTIO_QUEUE_NUM:

        ret = virtio_queue_get_num(vdev, vdev->queue_sel);

        break;

    case SYBORG_VIRTIO_QUEUE_SEL:

        ret = vdev->queue_sel;

        break;

    case SYBORG_VIRTIO_STATUS:

        ret = vdev->status;

        break;

    case SYBORG_VIRTIO_INT_ENABLE:

        ret = s->int_enable;

        break;

    case SYBORG_VIRTIO_INT_STATUS:

        ret = vdev->isr;

        break;

    default:

        BADF("Bad read offset 0x%x\n", (int)offset);

        return 0;

    }

    return ret;

}
