static void data_plane_blk_insert_notifier(Notifier *n, void *data)

{

    VirtIOBlockDataPlane *s = container_of(n, VirtIOBlockDataPlane,

                                           insert_notifier);

    assert(s->conf->conf.blk == data);

    data_plane_set_up_op_blockers(s);

}
