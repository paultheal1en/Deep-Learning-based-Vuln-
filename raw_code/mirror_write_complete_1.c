static void mirror_write_complete(void *opaque, int ret)

{

    MirrorOp *op = opaque;

    MirrorBlockJob *s = op->s;

    if (ret < 0) {

        BlockDriverState *source = s->common.bs;

        BlockErrorAction action;



        bdrv_set_dirty(source, op->sector_num, op->nb_sectors);

        action = mirror_error_action(s, false, -ret);

        if (action == BLOCK_ERROR_ACTION_REPORT && s->ret >= 0) {

            s->ret = ret;

        }

    }

    mirror_iteration_done(op, ret);

}
