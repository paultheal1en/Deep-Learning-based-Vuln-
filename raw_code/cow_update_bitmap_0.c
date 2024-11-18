static int cow_update_bitmap(BlockDriverState *bs, int64_t sector_num,

        int nb_sectors)

{

    int64_t bitnum = sector_num + sizeof(struct cow_header_v2) * 8;

    uint64_t offset = (bitnum / 8) & -BDRV_SECTOR_SIZE;

    bool first = true;

    int sector_bits;



    for ( ; nb_sectors;

            bitnum += sector_bits,

            nb_sectors -= sector_bits,

            offset += BDRV_SECTOR_SIZE) {

        int ret, set;

        uint8_t bitmap[BDRV_SECTOR_SIZE];



        bitnum &= BITS_PER_BITMAP_SECTOR - 1;

        sector_bits = MIN(nb_sectors, BITS_PER_BITMAP_SECTOR - bitnum);



        ret = bdrv_pread(bs->file, offset, &bitmap, sizeof(bitmap));

        if (ret < 0) {

            return ret;

        }



        /* Skip over any already set bits */

        set = cow_find_streak(bitmap, 1, bitnum, sector_bits);

        bitnum += set;

        sector_bits -= set;

        nb_sectors -= set;

        if (!sector_bits) {

            continue;

        }



        if (first) {

            ret = bdrv_flush(bs->file);

            if (ret < 0) {

                return ret;

            }

            first = false;

        }



        cow_set_bits(bitmap, bitnum, sector_bits);



        ret = bdrv_pwrite(bs->file, offset, &bitmap, sizeof(bitmap));

        if (ret < 0) {

            return ret;

        }

    }



    return 0;

}
