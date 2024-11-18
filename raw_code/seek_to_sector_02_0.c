static int64_t seek_to_sector(BDRVParallelsState *s, int64_t sector_num)

{

    uint32_t index, offset;



    index = sector_num / s->tracks;

    offset = sector_num % s->tracks;



    /* not allocated */

    if ((index >= s->catalog_size) || (s->catalog_bitmap[index] == 0))

        return -1;

    return (uint64_t)s->catalog_bitmap[index] * s->off_multiplier + offset;

}
