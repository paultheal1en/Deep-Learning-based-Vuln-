static void bdrv_move_feature_fields(BlockDriverState *bs_dest,

                                     BlockDriverState *bs_src)

{

    /* move some fields that need to stay attached to the device */

    bs_dest->open_flags         = bs_src->open_flags;



    /* dev info */

    bs_dest->dev_ops            = bs_src->dev_ops;

    bs_dest->dev_opaque         = bs_src->dev_opaque;

    bs_dest->dev                = bs_src->dev;

    bs_dest->buffer_alignment   = bs_src->buffer_alignment;

    bs_dest->copy_on_read       = bs_src->copy_on_read;



    bs_dest->enable_write_cache = bs_src->enable_write_cache;



    /* i/o timing parameters */

    bs_dest->slice_time         = bs_src->slice_time;

    bs_dest->slice_start        = bs_src->slice_start;

    bs_dest->slice_end          = bs_src->slice_end;

    bs_dest->io_limits          = bs_src->io_limits;

    bs_dest->io_base            = bs_src->io_base;

    bs_dest->throttled_reqs     = bs_src->throttled_reqs;

    bs_dest->block_timer        = bs_src->block_timer;

    bs_dest->io_limits_enabled  = bs_src->io_limits_enabled;



    /* geometry */

    bs_dest->cyls               = bs_src->cyls;

    bs_dest->heads              = bs_src->heads;

    bs_dest->secs               = bs_src->secs;

    bs_dest->translation        = bs_src->translation;



    /* r/w error */

    bs_dest->on_read_error      = bs_src->on_read_error;

    bs_dest->on_write_error     = bs_src->on_write_error;



    /* i/o status */

    bs_dest->iostatus_enabled   = bs_src->iostatus_enabled;

    bs_dest->iostatus           = bs_src->iostatus;



    /* dirty bitmap */

    bs_dest->dirty_count        = bs_src->dirty_count;

    bs_dest->dirty_bitmap       = bs_src->dirty_bitmap;



    /* job */

    bs_dest->in_use             = bs_src->in_use;

    bs_dest->job                = bs_src->job;



    /* keep the same entry in bdrv_states */

    pstrcpy(bs_dest->device_name, sizeof(bs_dest->device_name),

            bs_src->device_name);

    bs_dest->list = bs_src->list;

}
