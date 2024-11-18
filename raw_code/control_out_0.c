static void control_out(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtQueueElement elem;

    VirtIOSerial *vser;

    uint8_t *buf;

    size_t len;



    vser = VIRTIO_SERIAL(vdev);



    len = 0;

    buf = NULL;

    while (virtqueue_pop(vq, &elem)) {

        size_t cur_len;



        cur_len = iov_size(elem.out_sg, elem.out_num);

        /*

         * Allocate a new buf only if we didn't have one previously or

         * if the size of the buf differs

         */

        if (cur_len > len) {

            g_free(buf);



            buf = g_malloc(cur_len);

            len = cur_len;

        }

        iov_to_buf(elem.out_sg, elem.out_num, 0, buf, cur_len);



        handle_control_message(vser, buf, cur_len);

        virtqueue_push(vq, &elem, 0);

    }

    g_free(buf);

    virtio_notify(vdev, vq);

}
