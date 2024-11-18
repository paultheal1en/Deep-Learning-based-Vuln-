int coroutine_fn bdrv_co_pwrite_zeroes(BlockDriverState *bs,

                                       int64_t offset, int count,

                                       BdrvRequestFlags flags)

{

    trace_bdrv_co_pwrite_zeroes(bs, offset, count, flags);



    if (!(bs->open_flags & BDRV_O_UNMAP)) {

        flags &= ~BDRV_REQ_MAY_UNMAP;

    }



    return bdrv_co_pwritev(bs, offset, count, NULL,

                           BDRV_REQ_ZERO_WRITE | flags);

}
