static av_cold void init_mv_table(MVTable *tab)

{

    int i, x, y;



    tab->table_mv_index = av_malloc(sizeof(uint16_t) * 4096);

    /* mark all entries as not used */

    for(i=0;i<4096;i++)

        tab->table_mv_index[i] = tab->n;



    for(i=0;i<tab->n;i++) {

        x = tab->table_mvx[i];

        y = tab->table_mvy[i];

        tab->table_mv_index[(x << 6) | y] = i;

    }

}
