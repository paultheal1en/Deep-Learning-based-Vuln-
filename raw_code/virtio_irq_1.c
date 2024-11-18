void virtio_irq(VirtQueue *vq)

{

    trace_virtio_irq(vq);

    virtio_set_isr(vq->vdev, 0x1);

    virtio_notify_vector(vq->vdev, vq->vector);

}
