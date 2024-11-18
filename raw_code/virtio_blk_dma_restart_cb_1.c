static void virtio_blk_dma_restart_cb(void *opaque, int running,

                                      RunState state)

{

    VirtIOBlock *s = opaque;



    if (!running)

        return;



    if (!s->bh) {

        s->bh = qemu_bh_new(virtio_blk_dma_restart_bh, s);

        qemu_bh_schedule(s->bh);

    }

}
