int avcodec_get_pix_fmt_loss(int dst_pix_fmt, int src_pix_fmt,

                             int has_alpha)

{

    const PixFmtInfo *pf, *ps;

    int loss;



    ps = &pix_fmt_info[src_pix_fmt];

    pf = &pix_fmt_info[dst_pix_fmt];



    /* compute loss */

    loss = 0;

    pf = &pix_fmt_info[dst_pix_fmt];

    if (pf->depth < ps->depth)

        loss |= FF_LOSS_DEPTH;

    if (pf->x_chroma_shift >= ps->x_chroma_shift ||

        pf->y_chroma_shift >= ps->y_chroma_shift)

        loss |= FF_LOSS_RESOLUTION;

    switch(pf->color_type) {

    case FF_COLOR_RGB:

        if (ps->color_type != FF_COLOR_RGB &&

            ps->color_type != FF_COLOR_GRAY)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_GRAY:

        if (ps->color_type != FF_COLOR_GRAY)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_YUV:

        if (ps->color_type != FF_COLOR_YUV)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_YUV_JPEG:

        if (ps->color_type != FF_COLOR_YUV_JPEG &&

            ps->color_type != FF_COLOR_YUV)

            loss |= FF_LOSS_COLORSPACE;

        break;

    default:

        /* fail safe test */

        if (ps->color_type != pf->color_type)

            loss |= FF_LOSS_COLORSPACE;

        break;

    }

    if (pf->color_type == FF_COLOR_GRAY &&

        ps->color_type != FF_COLOR_GRAY)

        loss |= FF_LOSS_CHROMA;

    if (!pf->is_alpha && (ps->is_alpha && has_alpha))

        loss |= FF_LOSS_ALPHA;

    if (pf->is_paletted && (!ps->is_paletted && ps->color_type != FF_COLOR_GRAY))

        loss |= FF_LOSS_COLORQUANT;

    return loss;

}
