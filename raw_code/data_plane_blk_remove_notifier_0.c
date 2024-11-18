static void data_plane_blk_remove_notifier(Notifier *n, void *data)

{

    VirtIOBlockDataPlane *s = container_of(n, VirtIOBlockDataPlane,

                                           remove_notifier);

    assert(s->conf->conf.blk == data);

    data_plane_remove_op_blockers(s);

}
