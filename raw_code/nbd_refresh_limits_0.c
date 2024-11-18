static void nbd_refresh_limits(BlockDriverState *bs, Error **errp)

{

    bs->bl.max_pdiscard = NBD_MAX_BUFFER_SIZE;

    bs->bl.max_pwrite_zeroes = NBD_MAX_BUFFER_SIZE;

    bs->bl.max_transfer = NBD_MAX_BUFFER_SIZE;

}
