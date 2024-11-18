static int virtio_blk_load_device(VirtIODevice *vdev, QEMUFile *f,

                                  int version_id)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);



    while (qemu_get_sbyte(f)) {

        VirtIOBlockReq *req = virtio_blk_alloc_request(s);

        qemu_get_buffer(f, (unsigned char *)req->elem,

                        sizeof(VirtQueueElement));

        req->next = s->rq;

        s->rq = req;



        virtqueue_map_sg(req->elem->in_sg, req->elem->in_addr,

            req->elem->in_num, 1);

        virtqueue_map_sg(req->elem->out_sg, req->elem->out_addr,

            req->elem->out_num, 0);

    }



    return 0;

}
