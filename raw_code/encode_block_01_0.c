static void encode_block(MpegEncContext *s, int16_t *block, int n)

{

    int i, j, table_id;

    int component, dc, last_index, val, run;

    MJpegContext *m = s->mjpeg_ctx;



    /* DC coef */

    component = (n <= 3 ? 0 : (n&1) + 1);

    table_id = (n <= 3 ? 0 : 1);

    dc = block[0]; /* overflow is impossible */

    val = dc - s->last_dc[component];



    ff_mjpeg_encode_coef(m, table_id, val, 0);



    s->last_dc[component] = dc;



    /* AC coefs */



    run = 0;

    last_index = s->block_last_index[n];

    table_id |= 2;



    for(i=1;i<=last_index;i++) {

        j = s->intra_scantable.permutated[i];

        val = block[j];



        if (val == 0) {

            run++;

        } else {

            while (run >= 16) {

                ff_mjpeg_encode_code(m, table_id, 0xf0);

                run -= 16;

            }

            ff_mjpeg_encode_coef(m, table_id, val, run);

            run = 0;

        }

    }



    /* output EOB only if not already 64 values */

    if (last_index < 63 || run != 0)

        ff_mjpeg_encode_code(m, table_id, 0);

}
