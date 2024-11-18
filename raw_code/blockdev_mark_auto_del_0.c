void blockdev_mark_auto_del(BlockDriverState *bs)

{

    BlockBackend *blk = bs->blk;

    DriveInfo *dinfo = blk_legacy_dinfo(blk);



    if (dinfo && !dinfo->enable_auto_del) {

        return;

    }



    if (bs->job) {

        block_job_cancel(bs->job);

    }

    if (dinfo) {

        dinfo->auto_del = 1;

    }

}
