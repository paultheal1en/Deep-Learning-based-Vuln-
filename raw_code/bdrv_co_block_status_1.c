static int coroutine_fn bdrv_co_block_status(BlockDriverState *bs,

                                             bool want_zero,

                                             int64_t offset, int64_t bytes,

                                             int64_t *pnum, int64_t *map,

                                             BlockDriverState **file)

{

    int64_t total_size;

    int64_t n; /* bytes */

    int ret;

    int64_t local_map = 0;

    BlockDriverState *local_file = NULL;

    int64_t aligned_offset, aligned_bytes;

    uint32_t align;



    assert(pnum);

    *pnum = 0;

    total_size = bdrv_getlength(bs);

    if (total_size < 0) {

        ret = total_size;

        goto early_out;

    }



    if (offset >= total_size) {

        ret = BDRV_BLOCK_EOF;

        goto early_out;

    }

    if (!bytes) {

        ret = 0;

        goto early_out;

    }



    n = total_size - offset;

    if (n < bytes) {

        bytes = n;

    }





    if (!bs->drv->bdrv_co_get_block_status) {

        *pnum = bytes;

        ret = BDRV_BLOCK_DATA | BDRV_BLOCK_ALLOCATED;

        if (offset + bytes == total_size) {

            ret |= BDRV_BLOCK_EOF;

        }

        if (bs->drv->protocol_name) {

            ret |= BDRV_BLOCK_OFFSET_VALID;

            local_map = offset;

            local_file = bs;

        }

        goto early_out;

    }



    bdrv_inc_in_flight(bs);



    /* Round out to request_alignment boundaries */

    /* TODO: until we have a byte-based driver callback, we also have to

     * round out to sectors, even if that is bigger than request_alignment */

    align = MAX(bs->bl.request_alignment, BDRV_SECTOR_SIZE);

    aligned_offset = QEMU_ALIGN_DOWN(offset, align);

    aligned_bytes = ROUND_UP(offset + bytes, align) - aligned_offset;



    {

        int count; /* sectors */

        int64_t longret;



        assert(QEMU_IS_ALIGNED(aligned_offset | aligned_bytes,

                               BDRV_SECTOR_SIZE));

        /*

         * The contract allows us to return pnum smaller than bytes, even

         * if the next query would see the same status; we truncate the

         * request to avoid overflowing the driver's 32-bit interface.

         */

        longret = bs->drv->bdrv_co_get_block_status(

            bs, aligned_offset >> BDRV_SECTOR_BITS,

            MIN(INT_MAX, aligned_bytes) >> BDRV_SECTOR_BITS, &count,

            &local_file);

        if (longret < 0) {

            assert(INT_MIN <= longret);

            ret = longret;

            goto out;

        }

        if (longret & BDRV_BLOCK_OFFSET_VALID) {

            local_map = longret & BDRV_BLOCK_OFFSET_MASK;

        }

        ret = longret & ~BDRV_BLOCK_OFFSET_MASK;

        *pnum = count * BDRV_SECTOR_SIZE;

    }



    /*

     * The driver's result must be a multiple of request_alignment.

     * Clamp pnum and adjust map to original request.

     */

    assert(QEMU_IS_ALIGNED(*pnum, align) && align > offset - aligned_offset);

    *pnum -= offset - aligned_offset;

    if (*pnum > bytes) {

        *pnum = bytes;

    }

    if (ret & BDRV_BLOCK_OFFSET_VALID) {

        local_map += offset - aligned_offset;

    }



    if (ret & BDRV_BLOCK_RAW) {

        assert(ret & BDRV_BLOCK_OFFSET_VALID && local_file);

        ret = bdrv_co_block_status(local_file, want_zero, local_map,

                                   *pnum, pnum, &local_map, &local_file);

        goto out;

    }



    if (ret & (BDRV_BLOCK_DATA | BDRV_BLOCK_ZERO)) {

        ret |= BDRV_BLOCK_ALLOCATED;

    } else if (want_zero) {

        if (bdrv_unallocated_blocks_are_zero(bs)) {

            ret |= BDRV_BLOCK_ZERO;

        } else if (bs->backing) {

            BlockDriverState *bs2 = bs->backing->bs;

            int64_t size2 = bdrv_getlength(bs2);



            if (size2 >= 0 && offset >= size2) {

                ret |= BDRV_BLOCK_ZERO;

            }

        }

    }



    if (want_zero && local_file && local_file != bs &&

        (ret & BDRV_BLOCK_DATA) && !(ret & BDRV_BLOCK_ZERO) &&

        (ret & BDRV_BLOCK_OFFSET_VALID)) {

        int64_t file_pnum;

        int ret2;



        ret2 = bdrv_co_block_status(local_file, want_zero, local_map,

                                    *pnum, &file_pnum, NULL, NULL);

        if (ret2 >= 0) {

            /* Ignore errors.  This is just providing extra information, it

             * is useful but not necessary.

             */

            if (ret2 & BDRV_BLOCK_EOF &&

                (!file_pnum || ret2 & BDRV_BLOCK_ZERO)) {

                /*

                 * It is valid for the format block driver to read

                 * beyond the end of the underlying file's current

                 * size; such areas read as zero.

                 */

                ret |= BDRV_BLOCK_ZERO;

            } else {

                /* Limit request to the range reported by the protocol driver */

                *pnum = file_pnum;

                ret |= (ret2 & BDRV_BLOCK_ZERO);

            }

        }

    }



out:

    bdrv_dec_in_flight(bs);

    if (ret >= 0 && offset + *pnum == total_size) {

        ret |= BDRV_BLOCK_EOF;

    }

early_out:

    if (file) {

        *file = local_file;

    }

    if (map) {

        *map = local_map;

    }

    return ret;

}