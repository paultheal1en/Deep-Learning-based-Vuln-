static int raw_pread_aligned(BlockDriverState *bs, int64_t offset,

                     uint8_t *buf, int count)

{

    BDRVRawState *s = bs->opaque;

    int ret;



    ret = fd_open(bs);

    if (ret < 0)

        return ret;



    ret = pread(s->fd, buf, count, offset);

    if (ret == count)

        goto label__raw_read__success;



    /* Allow reads beyond the end (needed for pwrite) */

    if ((ret == 0) && bs->growable) {

        int64_t size = raw_getlength(bs);

        if (offset >= size) {

            memset(buf, 0, count);

            ret = count;

            goto label__raw_read__success;

        }

    }



    DEBUG_BLOCK_PRINT("raw_pread(%d:%s, %" PRId64 ", %p, %d) [%" PRId64

                      "] read failed %d : %d = %s\n",

                      s->fd, bs->filename, offset, buf, count,

                      bs->total_sectors, ret, errno, strerror(errno));



    /* Try harder for CDrom. */

    if (bs->type == BDRV_TYPE_CDROM) {

        ret = pread(s->fd, buf, count, offset);

        if (ret == count)

            goto label__raw_read__success;

        ret = pread(s->fd, buf, count, offset);

        if (ret == count)

            goto label__raw_read__success;



        DEBUG_BLOCK_PRINT("raw_pread(%d:%s, %" PRId64 ", %p, %d) [%" PRId64

                          "] retry read failed %d : %d = %s\n",

                          s->fd, bs->filename, offset, buf, count,

                          bs->total_sectors, ret, errno, strerror(errno));

    }



label__raw_read__success:



    return  (ret < 0) ? -errno : ret;

}
