int av_image_get_linesize(enum PixelFormat pix_fmt, int width, int plane)

{

    const AVPixFmtDescriptor *desc = &av_pix_fmt_descriptors[pix_fmt];

    int max_step     [4];       /* max pixel step for each plane */

    int max_step_comp[4];       /* the component for each plane which has the max pixel step */

    int s;



    if (desc->flags & PIX_FMT_BITSTREAM)

        return (width * (desc->comp[0].step_minus1+1) + 7) >> 3;



    av_image_fill_max_pixsteps(max_step, max_step_comp, desc);

    s = (max_step_comp[plane] == 1 || max_step_comp[plane] == 2) ? desc->log2_chroma_w : 0;

    return max_step[plane] * (((width + (1 << s) - 1)) >> s);

}
