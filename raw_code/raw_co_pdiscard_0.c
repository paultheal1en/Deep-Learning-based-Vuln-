static int coroutine_fn raw_co_pdiscard(BlockDriverState *bs,

                                        int64_t offset, int count)

{

    BDRVRawState *s = bs->opaque;

    if (offset > UINT64_MAX - s->offset) {

        return -EINVAL;

    }

    offset += s->offset;

    return bdrv_co_pdiscard(bs->file->bs, offset, count);

}
