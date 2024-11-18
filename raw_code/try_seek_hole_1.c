static int try_seek_hole(BlockDriverState *bs, off_t start, off_t *data,

                         off_t *hole)

{

#if defined SEEK_HOLE && defined SEEK_DATA

    BDRVRawState *s = bs->opaque;



    *hole = lseek(s->fd, start, SEEK_HOLE);

    if (*hole == -1) {

        return -errno;

    }



    if (*hole > start) {

        *data = start;

    } else {

        /* On a hole.  We need another syscall to find its end.  */

        *data = lseek(s->fd, start, SEEK_DATA);

        if (*data == -1) {

            *data = lseek(s->fd, 0, SEEK_END);

        }

    }



    return 0;

#else

    return -ENOTSUP;

#endif

}
