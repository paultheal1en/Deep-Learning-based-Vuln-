static void virtio_blk_save_device(VirtIODevice *vdev, QEMUFile *f)

{

    VirtIOBlock *s = VIRTIO_BLK(vdev);

    VirtIOBlockReq *req = s->rq;



    while (req) {

        qemu_put_sbyte(f, 1);

        qemu_put_buffer(f, (unsigned char *)req->elem,

                        sizeof(VirtQueueElement));

        req = req->next;

    }

    qemu_put_sbyte(f, 0);

}
