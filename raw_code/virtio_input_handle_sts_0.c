static void virtio_input_handle_sts(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOInputClass *vic = VIRTIO_INPUT_GET_CLASS(vdev);

    VirtIOInput *vinput = VIRTIO_INPUT(vdev);

    virtio_input_event event;

    VirtQueueElement elem;

    int len;



    while (virtqueue_pop(vinput->sts, &elem)) {

        memset(&event, 0, sizeof(event));

        len = iov_to_buf(elem.out_sg, elem.out_num,

                         0, &event, sizeof(event));

        if (vic->handle_status) {

            vic->handle_status(vinput, &event);

        }

        virtqueue_push(vinput->sts, &elem, len);

    }

    virtio_notify(vdev, vinput->sts);

}
