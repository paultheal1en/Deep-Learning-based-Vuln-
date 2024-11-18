int ff_draw_init(FFDrawContext *draw, enum PixelFormat format, unsigned flags)

{

    const AVPixFmtDescriptor *desc = &av_pix_fmt_descriptors[format];

    const AVComponentDescriptor *c;

    unsigned i, nb_planes = 0;

    int pixelstep[MAX_PLANES] = { 0 };



    if (!desc->name)

        return AVERROR(EINVAL);

    if (desc->flags & ~(PIX_FMT_PLANAR | PIX_FMT_RGB))

        return AVERROR(ENOSYS);

    for (i = 0; i < desc->nb_components; i++) {

        c = &desc->comp[i];

        /* for now, only 8-bits formats */

        if (c->depth_minus1 != 8 - 1)

            return AVERROR(ENOSYS);

        if (c->plane >= MAX_PLANES)

            return AVERROR(ENOSYS);

        /* strange interleaving */

        if (pixelstep[c->plane] != 0 &&

            pixelstep[c->plane] != c->step_minus1 + 1)

            return AVERROR(ENOSYS);

        pixelstep[c->plane] = c->step_minus1 + 1;

        if (pixelstep[c->plane] >= 8)

            return AVERROR(ENOSYS);

        nb_planes = FFMAX(nb_planes, c->plane + 1);

    }

    if ((desc->log2_chroma_w || desc->log2_chroma_h) && nb_planes < 3)

        return AVERROR(ENOSYS); /* exclude NV12 and NV21 */

    memset(draw, 0, sizeof(*draw));

    draw->desc      = desc;

    draw->format    = format;

    draw->nb_planes = nb_planes;

    memcpy(draw->pixelstep, pixelstep, sizeof(draw->pixelstep));

    if (nb_planes >= 3 && !(desc->flags & PIX_FMT_RGB)) {

        draw->hsub[1] = draw->hsub[2] = draw->hsub_max = desc->log2_chroma_w;

        draw->vsub[1] = draw->vsub[2] = draw->vsub_max = desc->log2_chroma_h;

    }

    for (i = 0; i < ((desc->nb_components - 1) | 1); i++)

        draw->comp_mask[desc->comp[i].plane] |=

            1 << (desc->comp[i].offset_plus1 - 1);

    return 0;

}
