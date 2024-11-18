static void pflash_update(pflash_t *pfl, int offset,

                          int size)

{

    int offset_end;

    if (pfl->bs) {

        offset_end = offset + size;

        /* round to sectors */

        offset = offset >> 9;

        offset_end = (offset_end + 511) >> 9;

        bdrv_write(pfl->bs, offset, pfl->storage + (offset << 9),

                   offset_end - offset);

    }

}
