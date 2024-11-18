static void mirror_read_complete(void *opaque, int ret)

{

    MirrorOp *op = opaque;

    MirrorBlockJob *s = op->s;



    aio_context_acquire(blk_get_aio_context(s->common.blk));

    if (ret < 0) {

        BlockErrorAction action;



        bdrv_set_dirty_bitmap(s->dirty_bitmap, op->sector_num, op->nb_sectors);

        action = mirror_error_action(s, true, -ret);

        if (action == BLOCK_ERROR_ACTION_REPORT && s->ret >= 0) {

            s->ret = ret;

        }



        mirror_iteration_done(op, ret);

    } else {

        blk_aio_pwritev(s->target, op->sector_num * BDRV_SECTOR_SIZE, &op->qiov,

                        0, mirror_write_complete, op);

    }

    aio_context_release(blk_get_aio_context(s->common.blk));

}
