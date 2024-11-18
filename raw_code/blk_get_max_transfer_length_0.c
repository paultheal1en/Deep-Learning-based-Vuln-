int blk_get_max_transfer_length(BlockBackend *blk)

{

    BlockDriverState *bs = blk_bs(blk);



    if (bs) {

        return bs->bl.max_transfer_length;

    } else {

        return 0;

    }

}
