static int check_image_pointers(uint8_t *data[4], enum AVPixelFormat pix_fmt,

                                const int linesizes[4])

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);

    int i;



    for (i = 0; i < 4; i++) {

        int plane = desc->comp[i].plane;

        if (!data[plane] || !linesizes[plane])

            return 0;

    }



    return 1;

}
