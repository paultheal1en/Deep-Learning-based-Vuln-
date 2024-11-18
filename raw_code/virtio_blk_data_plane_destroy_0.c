void virtio_blk_data_plane_destroy(VirtIOBlockDataPlane *s)

{

    if (!s) {

        return;

    }



    virtio_blk_data_plane_stop(s);

    g_free(s->batch_notify_vqs);

    qemu_bh_delete(s->bh);

    object_unref(OBJECT(s->iothread));

    g_free(s);

}
