static int gen_sub_bitmap(TeletextContext *ctx, AVSubtitleRect *sub_rect, vbi_page *page, int chop_top)

{

    int resx = page->columns * BITMAP_CHAR_WIDTH;

    int resy = (page->rows - chop_top) * BITMAP_CHAR_HEIGHT;

    uint8_t ci, cmax = 0;

    int ret;

    vbi_char *vc = page->text + (chop_top * page->columns);

    vbi_char *vcend = page->text + (page->rows * page->columns);



    for (; vc < vcend; vc++) {

        if (vc->opacity != VBI_TRANSPARENT_SPACE) {

            cmax = VBI_NB_COLORS;

            break;

        }

    }



    if (cmax == 0) {

        av_log(ctx, AV_LOG_DEBUG, "dropping empty page %3x\n", page->pgno);

        sub_rect->type = SUBTITLE_NONE;

        return 0;

    }



    if ((ret = avpicture_alloc(&sub_rect->pict, AV_PIX_FMT_PAL8, resx, resy)) < 0)

        return ret;

    // Yes, we want to allocate the palette on our own because AVSubtitle works this way

    sub_rect->pict.data[1] = NULL;



    vbi_draw_vt_page_region(page, VBI_PIXFMT_PAL8,

                            sub_rect->pict.data[0], sub_rect->pict.linesize[0],

                            0, chop_top, page->columns, page->rows - chop_top,

                            /*reveal*/ 1, /*flash*/ 1);



    fix_transparency(ctx, sub_rect, page, chop_top, cmax, resx, resy);

    sub_rect->x = ctx->x_offset;

    sub_rect->y = ctx->y_offset + chop_top * BITMAP_CHAR_HEIGHT;

    sub_rect->w = resx;

    sub_rect->h = resy;

    sub_rect->nb_colors = (int)cmax + 1;

    sub_rect->pict.data[1] = av_mallocz(AVPALETTE_SIZE);

    if (!sub_rect->pict.data[1]) {

        av_freep(&sub_rect->pict.data[0]);

        return AVERROR(ENOMEM);

    }

    for (ci = 0; ci < cmax; ci++) {

        int r, g, b, a;



        r = VBI_R(page->color_map[ci]);

        g = VBI_G(page->color_map[ci]);

        b = VBI_B(page->color_map[ci]);

        a = VBI_A(page->color_map[ci]);

        ((uint32_t *)sub_rect->pict.data[1])[ci] = RGBA(r, g, b, a);

        av_dlog(ctx, "palette %0x\n", ((uint32_t *)sub_rect->pict.data[1])[ci]);

    }

    ((uint32_t *)sub_rect->pict.data[1])[cmax] = RGBA(0, 0, 0, 0);

    sub_rect->type = SUBTITLE_BITMAP;

    return 0;

}
