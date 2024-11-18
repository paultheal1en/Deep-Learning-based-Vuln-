static int walk_memory_regions_1(struct walk_memory_regions_data *data,

                                 abi_ulong base, int level, void **lp)

{

    abi_ulong pa;

    int i, rc;



    if (*lp == NULL) {

        return walk_memory_regions_end(data, base, 0);

    }



    if (level == 0) {

        PageDesc *pd = *lp;



        for (i = 0; i < L2_SIZE; ++i) {

            int prot = pd[i].flags;



            pa = base | (i << TARGET_PAGE_BITS);

            if (prot != data->prot) {

                rc = walk_memory_regions_end(data, pa, prot);

                if (rc != 0) {

                    return rc;

                }

            }

        }

    } else {

        void **pp = *lp;



        for (i = 0; i < L2_SIZE; ++i) {

            pa = base | ((abi_ulong)i <<

                (TARGET_PAGE_BITS + L2_BITS * level));

            rc = walk_memory_regions_1(data, pa, level - 1, pp + i);

            if (rc != 0) {

                return rc;

            }

        }

    }



    return 0;

}
