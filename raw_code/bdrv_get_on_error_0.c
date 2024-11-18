BlockdevOnError bdrv_get_on_error(BlockDriverState *bs, bool is_read)

{

    return is_read ? bs->on_read_error : bs->on_write_error;

}
