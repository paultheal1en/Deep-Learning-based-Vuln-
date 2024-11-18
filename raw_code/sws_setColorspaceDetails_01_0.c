int sws_setColorspaceDetails(struct SwsContext *c, const int inv_table[4],

                             int srcRange, const int table[4], int dstRange,

                             int brightness, int contrast, int saturation)

{

    const AVPixFmtDescriptor *desc_dst = av_pix_fmt_desc_get(c->dstFormat);

    const AVPixFmtDescriptor *desc_src = av_pix_fmt_desc_get(c->srcFormat);

    memcpy(c->srcColorspaceTable, inv_table, sizeof(int) * 4);

    memcpy(c->dstColorspaceTable, table, sizeof(int) * 4);



    c->brightness = brightness;

    c->contrast   = contrast;

    c->saturation = saturation;

    c->srcRange   = srcRange;

    c->dstRange   = dstRange;

    if (isYUV(c->dstFormat) || isGray(c->dstFormat))

        return -1;



    c->dstFormatBpp = av_get_bits_per_pixel(desc_dst);

    c->srcFormatBpp = av_get_bits_per_pixel(desc_src);



    ff_yuv2rgb_c_init_tables(c, inv_table, srcRange, brightness,

                             contrast, saturation);

    // FIXME factorize



    if (HAVE_ALTIVEC && av_get_cpu_flags() & AV_CPU_FLAG_ALTIVEC)

        ff_yuv2rgb_init_tables_altivec(c, inv_table, brightness,

                                       contrast, saturation);

    return 0;

}
