int bdrv_is_allocated_above(BlockDriverState *top,

                            BlockDriverState *base,

                            int64_t sector_num,

                            int nb_sectors, int *pnum)

{

    BlockDriverState *intermediate;

    int ret, n = nb_sectors;



    intermediate = top;

    while (intermediate && intermediate != base) {

        int pnum_inter;

        ret = bdrv_is_allocated(intermediate, sector_num, nb_sectors,

                                &pnum_inter);

        if (ret < 0) {

            return ret;

        } else if (ret) {

            *pnum = pnum_inter;

            return 1;

        }



        /*

         * [sector_num, nb_sectors] is unallocated on top but intermediate

         * might have

         *

         * [sector_num+x, nr_sectors] allocated.

         */

        if (n > pnum_inter &&

            (intermediate == top ||

             sector_num + pnum_inter < intermediate->total_sectors)) {

            n = pnum_inter;

        }



        intermediate = intermediate->backing_hd;

    }



    *pnum = n;

    return 0;

}
