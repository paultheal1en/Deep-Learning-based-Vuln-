void virtio_blk_data_plane_destroy(VirtIOBlockDataPlane *s)

{

    if (!s) {

        return;

    }



    virtio_blk_data_plane_stop(s);

    blk_op_unblock_all(s->conf->conf.blk, s->blocker);

    error_free(s->blocker);

    object_unref(OBJECT(s->iothread));

    qemu_bh_delete(s->bh);

    g_free(s);

}
