int av_image_alloc(uint8_t *pointers[4], int linesizes[4],

                   int w, int h, enum PixelFormat pix_fmt, int align)

{

    int i, ret;

    uint8_t *buf;



    if ((ret = av_image_check_size(w, h, 0, NULL)) < 0)

        return ret;

    if ((ret = av_image_fill_linesizes(linesizes, pix_fmt, w)) < 0)

        return ret;



    for (i = 0; i < 4; i++)

        linesizes[i] = FFALIGN(linesizes[i], align);



    if ((ret = av_image_fill_pointers(pointers, pix_fmt, h, NULL, linesizes)) < 0)

        return ret;

    buf = av_malloc(ret + align);

    if (!buf)

        return AVERROR(ENOMEM);

    if ((ret = av_image_fill_pointers(pointers, pix_fmt, h, buf, linesizes)) < 0) {

        av_free(buf);

        return ret;

    }

    if (av_pix_fmt_descriptors[pix_fmt].flags & PIX_FMT_PAL)

        ff_set_systematic_pal2((uint32_t*)pointers[1], pix_fmt);



    return ret;

}
