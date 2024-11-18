static int save_subtitle_set(AVCodecContext *avctx, AVSubtitle *sub, int *got_output)

{

    DVBSubContext *ctx = avctx->priv_data;

    DVBSubRegionDisplay *display;

    DVBSubDisplayDefinition *display_def = ctx->display_definition;

    DVBSubRegion *region;

    AVSubtitleRect *rect;

    DVBSubCLUT *clut;

    uint32_t *clut_table;

    int i;

    int offset_x=0, offset_y=0;

    int ret = 0;





    if (display_def) {

        offset_x = display_def->x;

        offset_y = display_def->y;

    }



    /* Not touching AVSubtitles again*/

    if(sub->num_rects) {

        avpriv_request_sample(ctx, "Different Version of Segment asked Twice");

        return AVERROR_PATCHWELCOME;

    }

    for (display = ctx->display_list; display; display = display->next) {

        region = get_region(ctx, display->region_id);

        if (region && region->dirty)

            sub->num_rects++;

    }



    if(ctx->compute_edt == 0) {

        sub->end_display_time = ctx->time_out * 1000;

        *got_output = 1;

    } else if (ctx->prev_start != AV_NOPTS_VALUE) {

        sub->end_display_time = av_rescale_q((sub->pts - ctx->prev_start ), AV_TIME_BASE_Q, (AVRational){ 1, 1000 }) - 1;

        *got_output = 1;

    }

    if (sub->num_rects > 0) {



        sub->rects = av_mallocz_array(sizeof(*sub->rects), sub->num_rects);

        if (!sub->rects) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        for (i = 0; i < sub->num_rects; i++) {

            sub->rects[i] = av_mallocz(sizeof(*sub->rects[i]));

            if (!sub->rects[i]) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

        }



        i = 0;



        for (display = ctx->display_list; display; display = display->next) {

            region = get_region(ctx, display->region_id);



            if (!region)

                continue;



            if (!region->dirty)

                continue;



            rect = sub->rects[i];

            rect->x = display->x_pos + offset_x;

            rect->y = display->y_pos + offset_y;

            rect->w = region->width;

            rect->h = region->height;

            rect->nb_colors = (1 << region->depth);

            rect->type      = SUBTITLE_BITMAP;

            rect->linesize[0] = region->width;



            clut = get_clut(ctx, region->clut);



            if (!clut)

                clut = &default_clut;



            switch (region->depth) {

            case 2:

                clut_table = clut->clut4;

                break;

            case 8:

                clut_table = clut->clut256;

                break;

            case 4:

            default:

                clut_table = clut->clut16;

                break;

            }



            rect->data[1] = av_mallocz(AVPALETTE_SIZE);

            if (!rect->data[1]) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

            memcpy(rect->data[1], clut_table, (1 << region->depth) * sizeof(uint32_t));



            rect->data[0] = av_malloc(region->buf_size);

            if (!rect->data[0]) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }



            memcpy(rect->data[0], region->pbuf, region->buf_size);



            if ((clut == &default_clut && ctx->compute_clut == -1) || ctx->compute_clut == 1)

                compute_default_clut(rect, rect->w, rect->h);



#if FF_API_AVPICTURE

FF_DISABLE_DEPRECATION_WARNINGS

{

            int j;

            for (j = 0; j < 4; j++) {

                rect->pict.data[j] = rect->data[j];

                rect->pict.linesize[j] = rect->linesize[j];

            }

}

FF_ENABLE_DEPRECATION_WARNINGS

#endif



            i++;

        }

    }



    return 0;

fail:

    if (sub->rects) {

        for(i=0; i<sub->num_rects; i++) {

            rect = sub->rects[i];

            if (rect) {

                av_freep(&rect->data[0]);

                av_freep(&rect->data[1]);

            }

            av_freep(&sub->rects[i]);

        }

        av_freep(&sub->rects);

    }

    sub->num_rects = 0;

    return ret;

}
