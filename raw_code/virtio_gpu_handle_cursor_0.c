static void virtio_gpu_handle_cursor(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOGPU *g = VIRTIO_GPU(vdev);

    VirtQueueElement elem;

    size_t s;

    struct virtio_gpu_update_cursor cursor_info;



    if (!virtio_queue_ready(vq)) {

        return;

    }

    while (virtqueue_pop(vq, &elem)) {

        s = iov_to_buf(elem.out_sg, elem.out_num, 0,

                       &cursor_info, sizeof(cursor_info));

        if (s != sizeof(cursor_info)) {

            qemu_log_mask(LOG_GUEST_ERROR,

                          "%s: cursor size incorrect %zu vs %zu\n",

                          __func__, s, sizeof(cursor_info));

        } else {

            update_cursor(g, &cursor_info);

        }

        virtqueue_push(vq, &elem, 0);

        virtio_notify(vdev, vq);

    }

}
