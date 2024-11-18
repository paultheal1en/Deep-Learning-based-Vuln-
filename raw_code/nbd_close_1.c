static void nbd_close(BlockDriverState *bs)
{
    nbd_teardown_connection(bs);
}