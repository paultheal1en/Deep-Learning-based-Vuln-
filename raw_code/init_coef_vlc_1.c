static void init_coef_vlc(VLC *vlc, 

                          uint16_t **prun_table, uint16_t **plevel_table,

                          const CoefVLCTable *vlc_table)

{

    int n = vlc_table->n;

    const uint8_t *table_bits = vlc_table->huffbits;

    const uint32_t *table_codes = vlc_table->huffcodes;

    const uint16_t *levels_table = vlc_table->levels;

    uint16_t *run_table, *level_table;

    const uint16_t *p;

    int i, l, j, level;



    init_vlc(vlc, 9, n, table_bits, 1, 1, table_codes, 4, 4);



    run_table = av_malloc(n * sizeof(uint16_t));

    level_table = av_malloc(n * sizeof(uint16_t));

    p = levels_table;

    i = 2;

    level = 1;

    while (i < n) {

        l = *p++;

        for(j=0;j<l;j++) {

            run_table[i] = j;

            level_table[i] = level;

            i++;

        }

        level++;

    }

    *prun_table = run_table;

    *plevel_table = level_table;

}
