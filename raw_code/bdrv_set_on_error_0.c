void bdrv_set_on_error(BlockDriverState *bs, BlockdevOnError on_read_error,

                       BlockdevOnError on_write_error)

{

    bs->on_read_error = on_read_error;

    bs->on_write_error = on_write_error;

}
