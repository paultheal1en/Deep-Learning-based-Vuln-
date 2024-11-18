static int64_t allocate_cluster(BlockDriverState *bs, int64_t sector_num)

{

    BDRVParallelsState *s = bs->opaque;

    uint32_t idx, offset, tmp;

    int64_t pos;

    int ret;



    idx = sector_num / s->tracks;

    offset = sector_num % s->tracks;



    if (idx >= s->catalog_size) {

        return -EINVAL;

    }

    if (s->catalog_bitmap[idx] != 0) {

        return (uint64_t)s->catalog_bitmap[idx] * s->off_multiplier + offset;

    }



    pos = bdrv_getlength(bs->file) >> BDRV_SECTOR_BITS;

    if (s->has_truncate) {

        ret = bdrv_truncate(bs->file, (pos + s->tracks) << BDRV_SECTOR_BITS);

    } else {

        ret = bdrv_write_zeroes(bs->file, pos, s->tracks, 0);

    }

    if (ret < 0) {

        return ret;

    }



    s->catalog_bitmap[idx] = pos / s->off_multiplier;



    tmp = cpu_to_le32(s->catalog_bitmap[idx]);



    ret = bdrv_pwrite(bs->file,

            sizeof(ParallelsHeader) + idx * sizeof(tmp), &tmp, sizeof(tmp));

    if (ret < 0) {

        s->catalog_bitmap[idx] = 0;

        return ret;

    }

    return (uint64_t)s->catalog_bitmap[idx] * s->off_multiplier + offset;

}
