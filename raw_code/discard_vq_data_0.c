static void discard_vq_data(VirtQueue *vq, VirtIODevice *vdev)

{

    VirtQueueElement elem;



    if (!virtio_queue_ready(vq)) {

        return;

    }

    while (virtqueue_pop(vq, &elem)) {

        virtqueue_push(vq, &elem, 0);

    }

    virtio_notify(vdev, vq);

}
