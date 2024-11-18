static int do_write_compressed(BlockBackend *blk, char *buf, int64_t offset,

                               int64_t count, int64_t *total)

{

    int ret;



    if (count >> 9 > INT_MAX) {

        return -ERANGE;

    }



    ret = blk_write_compressed(blk, offset >> 9, (uint8_t *)buf, count >> 9);

    if (ret < 0) {

        return ret;

    }

    *total = count;

    return 1;

}
