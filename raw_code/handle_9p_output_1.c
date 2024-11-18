static void handle_9p_output(VirtIODevice *vdev, VirtQueue *vq)

{

    V9fsVirtioState *v = (V9fsVirtioState *)vdev;

    V9fsState *s = &v->state;

    V9fsPDU *pdu;

    ssize_t len;



    while ((pdu = pdu_alloc(s))) {

        struct {

            uint32_t size_le;

            uint8_t id;

            uint16_t tag_le;

        } QEMU_PACKED out;

        VirtQueueElement *elem;



        elem = virtqueue_pop(vq, sizeof(VirtQueueElement));

        if (!elem) {

            pdu_free(pdu);

            break;

        }



        BUG_ON(elem->out_num == 0 || elem->in_num == 0);

        QEMU_BUILD_BUG_ON(sizeof(out) != 7);



        v->elems[pdu->idx] = elem;

        len = iov_to_buf(elem->out_sg, elem->out_num, 0,

                         &out, sizeof(out));

        BUG_ON(len != sizeof(out));



        pdu->size = le32_to_cpu(out.size_le);



        pdu->id = out.id;

        pdu->tag = le16_to_cpu(out.tag_le);



        qemu_co_queue_init(&pdu->complete);

        pdu_submit(pdu);

    }

}
