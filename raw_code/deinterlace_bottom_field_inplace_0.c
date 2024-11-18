static void deinterlace_bottom_field_inplace(uint8_t *src1, int src_wrap,

                                             int width, int height)

{

    uint8_t *src_m1, *src_0, *src_p1, *src_p2;

    int y;

    uint8_t *buf;

    buf = av_malloc(width);



    src_m1 = src1;

    memcpy(buf,src_m1,width);

    src_0=&src_m1[src_wrap];

    src_p1=&src_0[src_wrap];

    src_p2=&src_p1[src_wrap];

    for(y=0;y<(height-2);y+=2) {

        deinterlace_line_inplace(buf,src_m1,src_0,src_p1,src_p2,width);

        src_m1 = src_p1;

        src_0 = src_p2;

        src_p1 += 2*src_wrap;

        src_p2 += 2*src_wrap;

    }

    /* do last line */

    deinterlace_line_inplace(buf,src_m1,src_0,src_0,src_0,width);

    av_free(buf);

}
