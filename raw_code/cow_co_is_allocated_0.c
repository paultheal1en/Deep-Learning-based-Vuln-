static int coroutine_fn cow_co_is_allocated(BlockDriverState *bs,

        int64_t sector_num, int nb_sectors, int *num_same)

{

    int64_t bitnum = sector_num + sizeof(struct cow_header_v2) * 8;

    uint64_t offset = (bitnum / 8) & -BDRV_SECTOR_SIZE;

    bool first = true;

    int changed = 0, same = 0;



    do {

        int ret;

        uint8_t bitmap[BDRV_SECTOR_SIZE];



        bitnum &= BITS_PER_BITMAP_SECTOR - 1;

        int sector_bits = MIN(nb_sectors, BITS_PER_BITMAP_SECTOR - bitnum);



        ret = bdrv_pread(bs->file, offset, &bitmap, sizeof(bitmap));

        if (ret < 0) {

            return ret;

        }



        if (first) {

            changed = cow_test_bit(bitnum, bitmap);

            first = false;

        }



        same += cow_find_streak(bitmap, changed, bitnum, nb_sectors);



        bitnum += sector_bits;

        nb_sectors -= sector_bits;

        offset += BDRV_SECTOR_SIZE;

    } while (nb_sectors);



    *num_same = same;

    return changed;

}
