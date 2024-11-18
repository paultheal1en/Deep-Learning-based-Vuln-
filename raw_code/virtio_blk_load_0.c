static int virtio_blk_load(QEMUFile *f, void *opaque, int version_id)

{

    VirtIOBlock *s = opaque;



    if (version_id != 2)

        return -EINVAL;



    virtio_load(&s->vdev, f);

    while (qemu_get_sbyte(f)) {

        VirtIOBlockReq *req = virtio_blk_alloc_request(s);

        qemu_get_buffer(f, (unsigned char*)&req->elem, sizeof(req->elem));

        req->next = s->rq;

        s->rq = req->next;

    }



    return 0;

}
