static void bmds_set_aio_inflight(BlkMigDevState *bmds, int64_t sector_num,

                             int nb_sectors, int set)

{

    int64_t start, end;

    unsigned long val, idx, bit;



    start = sector_num / BDRV_SECTORS_PER_DIRTY_CHUNK;

    end = (sector_num + nb_sectors - 1) / BDRV_SECTORS_PER_DIRTY_CHUNK;



    for (; start <= end; start++) {

        idx = start / (sizeof(unsigned long) * 8);

        bit = start % (sizeof(unsigned long) * 8);

        val = bmds->aio_bitmap[idx];

        if (set) {

            val |= 1UL << bit;

        } else {

            val &= ~(1UL << bit);

        }

        bmds->aio_bitmap[idx] = val;

    }

}
