static void do_multiwrite(BlockDriverState *bs, BlockRequest *blkreq,

    int num_writes)

{

    int i, ret;

    ret = bdrv_aio_multiwrite(bs, blkreq, num_writes);



    if (ret != 0) {

        for (i = 0; i < num_writes; i++) {

            if (blkreq[i].error) {

                virtio_blk_rw_complete(blkreq[i].opaque, -EIO);

            }

        }

    }

}
