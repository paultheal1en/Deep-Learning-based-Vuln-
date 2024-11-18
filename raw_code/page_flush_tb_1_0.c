static void page_flush_tb_1(int level, void **lp)

{

    int i;



    if (*lp == NULL) {

        return;

    }

    if (level == 0) {

        PageDesc *pd = *lp;



        for (i = 0; i < L2_SIZE; ++i) {

            pd[i].first_tb = NULL;

            invalidate_page_bitmap(pd + i);

        }

    } else {

        void **pp = *lp;



        for (i = 0; i < L2_SIZE; ++i) {

            page_flush_tb_1(level - 1, pp + i);

        }

    }

}
