void virtio_notify(VirtIODevice *vdev, VirtQueue *vq)

{

    /* Always notify when queue is empty */

    if ((vq->inuse || vring_avail_idx(vq) != vq->last_avail_idx) &&

        (vring_avail_flags(vq) & VRING_AVAIL_F_NO_INTERRUPT))

        return;



    vdev->isr |= 0x01;

    virtio_update_irq(vdev);

}
