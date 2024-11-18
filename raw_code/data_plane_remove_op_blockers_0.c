static void data_plane_remove_op_blockers(VirtIOBlockDataPlane *s)

{

    if (s->blocker) {

        blk_op_unblock_all(s->conf->conf.blk, s->blocker);

        error_free(s->blocker);

        s->blocker = NULL;

    }

}
