static void do_commit(void)

{

    int i;



    for (i = 0; i < MAX_DISKS; i++) {

        if (bs_table[i])

            bdrv_commit(bs_table[i]);

    }

}
