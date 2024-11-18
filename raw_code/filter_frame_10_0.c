static int filter_frame(AVFilterLink *inlink, AVFrame *inpicref)

{

    AVFilterContext *ctx  = inlink->dst;

    Stereo3DContext *s = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    AVFrame *out, *oleft, *oright, *ileft, *iright;

    int out_off_left[4], out_off_right[4];

    int i;



    if (s->in.format == s->out.format)

        return ff_filter_frame(outlink, inpicref);



    switch (s->in.format) {

    case ALTERNATING_LR:

    case ALTERNATING_RL:

        if (!s->prev) {

            s->prev = inpicref;

            return 0;

        }

        ileft  = s->prev;

        iright = inpicref;

        if (s->in.format == ALTERNATING_RL)

            FFSWAP(AVFrame *, ileft, iright);

        break;

    default:

        ileft = iright = inpicref;

    };



    if ((s->out.format == ALTERNATING_LR ||

         s->out.format == ALTERNATING_RL) &&

        (s->in.format == SIDE_BY_SIDE_LR ||

         s->in.format == SIDE_BY_SIDE_RL ||

         s->in.format == SIDE_BY_SIDE_2_LR ||

         s->in.format == SIDE_BY_SIDE_2_RL ||

         s->in.format == ABOVE_BELOW_LR ||

         s->in.format == ABOVE_BELOW_RL ||

         s->in.format == ABOVE_BELOW_2_LR ||

         s->in.format == ABOVE_BELOW_2_RL)) {

        oright = av_frame_clone(inpicref);

        oleft  = av_frame_clone(inpicref);

        if (!oright || !oleft) {

            av_frame_free(&oright);

            av_frame_free(&oleft);

            av_frame_free(&s->prev);

            av_frame_free(&inpicref);

            return AVERROR(ENOMEM);

        }

    } else if ((s->out.format == MONO_L ||

                s->out.format == MONO_R) &&

        (s->in.format == SIDE_BY_SIDE_LR ||

         s->in.format == SIDE_BY_SIDE_RL ||

         s->in.format == SIDE_BY_SIDE_2_LR ||

         s->in.format == SIDE_BY_SIDE_2_RL ||

         s->in.format == ABOVE_BELOW_LR ||

         s->in.format == ABOVE_BELOW_RL ||

         s->in.format == ABOVE_BELOW_2_LR ||

         s->in.format == ABOVE_BELOW_2_RL)) {

        out = oleft = oright = av_frame_clone(inpicref);

        if (!out) {

            av_frame_free(&s->prev);

            av_frame_free(&inpicref);

            return AVERROR(ENOMEM);

        }

    } else {

        out = oleft = oright = ff_get_video_buffer(outlink, outlink->w, outlink->h);

        if (!out) {

            av_frame_free(&s->prev);

            av_frame_free(&inpicref);

            return AVERROR(ENOMEM);

        }

        av_frame_copy_props(out, inpicref);



        if (s->out.format == ALTERNATING_LR ||

            s->out.format == ALTERNATING_RL) {

            oright = ff_get_video_buffer(outlink, outlink->w, outlink->h);

            if (!oright) {

                av_frame_free(&oleft);

                av_frame_free(&s->prev);

                av_frame_free(&inpicref);

                return AVERROR(ENOMEM);

            }

            av_frame_copy_props(oright, inpicref);

        }

    }



    for (i = 0; i < 4; i++) {

        int hsub = i == 1 || i == 2 ? s->hsub : 0;

        int vsub = i == 1 || i == 2 ? s->vsub : 0;

        s->in_off_left[i]   = (FF_CEIL_RSHIFT(s->in.row_left,   vsub) + s->in.off_lstep)  * ileft->linesize[i]  + FF_CEIL_RSHIFT(s->in.off_left   * s->pixstep[i], hsub);

        s->in_off_right[i]  = (FF_CEIL_RSHIFT(s->in.row_right,  vsub) + s->in.off_rstep)  * iright->linesize[i] + FF_CEIL_RSHIFT(s->in.off_right  * s->pixstep[i], hsub);

        out_off_left[i]  = (FF_CEIL_RSHIFT(s->out.row_left,  vsub) + s->out.off_lstep) * oleft->linesize[i]  + FF_CEIL_RSHIFT(s->out.off_left  * s->pixstep[i], hsub);

        out_off_right[i] = (FF_CEIL_RSHIFT(s->out.row_right, vsub) + s->out.off_rstep) * oright->linesize[i] + FF_CEIL_RSHIFT(s->out.off_right * s->pixstep[i], hsub);

    }



    switch (s->out.format) {

    case ALTERNATING_LR:

    case ALTERNATING_RL:

        switch (s->in.format) {

        case ABOVE_BELOW_LR:

        case ABOVE_BELOW_RL:

        case ABOVE_BELOW_2_LR:

        case ABOVE_BELOW_2_RL:

        case SIDE_BY_SIDE_LR:

        case SIDE_BY_SIDE_RL:

        case SIDE_BY_SIDE_2_LR:

        case SIDE_BY_SIDE_2_RL:

            oleft->width   = outlink->w;

            oright->width  = outlink->w;

            oleft->height  = outlink->h;

            oright->height = outlink->h;



            for (i = 0; i < s->nb_planes; i++) {

                oleft->data[i]  += s->in_off_left[i];

                oright->data[i] += s->in_off_right[i];

            }

            break;

        default:

            goto copy;

            break;

        }

        break;

    case HDMI:

        for (i = 0; i < s->nb_planes; i++) {

            int j, h = s->height >> ((i == 1 || i == 2) ? s->vsub : 0);

            int b = (s->blanks) >> ((i == 1 || i == 2) ? s->vsub : 0);



            for (j = h; j < h + b; j++)

                memset(oleft->data[i] + j * s->linesize[i], 0, s->linesize[i]);

        }

    case SIDE_BY_SIDE_LR:

    case SIDE_BY_SIDE_RL:

    case SIDE_BY_SIDE_2_LR:

    case SIDE_BY_SIDE_2_RL:

    case ABOVE_BELOW_LR:

    case ABOVE_BELOW_RL:

    case ABOVE_BELOW_2_LR:

    case ABOVE_BELOW_2_RL:

    case INTERLEAVE_ROWS_LR:

    case INTERLEAVE_ROWS_RL:

copy:

        for (i = 0; i < s->nb_planes; i++) {

            av_image_copy_plane(oleft->data[i] + out_off_left[i],

                                oleft->linesize[i] * s->out.row_step,

                                ileft->data[i] + s->in_off_left[i],

                                ileft->linesize[i] * s->in.row_step,

                                s->linesize[i], s->pheight[i]);

            av_image_copy_plane(oright->data[i] + out_off_right[i],

                                oright->linesize[i] * s->out.row_step,

                                iright->data[i] + s->in_off_right[i],

                                iright->linesize[i] * s->in.row_step,

                                s->linesize[i], s->pheight[i]);

        }

        break;

    case MONO_L:

        iright = ileft;

    case MONO_R:

        switch (s->in.format) {

        case ABOVE_BELOW_LR:

        case ABOVE_BELOW_RL:

        case ABOVE_BELOW_2_LR:

        case ABOVE_BELOW_2_RL:

        case SIDE_BY_SIDE_LR:

        case SIDE_BY_SIDE_RL:

        case SIDE_BY_SIDE_2_LR:

        case SIDE_BY_SIDE_2_RL:

            out->width  = outlink->w;

            out->height = outlink->h;



            for (i = 0; i < s->nb_planes; i++) {

                out->data[i] += s->in_off_left[i];

            }

            break;

        default:

            for (i = 0; i < s->nb_planes; i++) {

                av_image_copy_plane(out->data[i], out->linesize[i],

                                    iright->data[i] + s->in_off_left[i],

                                    iright->linesize[i] * s->in.row_step,

                                    s->linesize[i], s->pheight[i]);

            }

            break;

        }

        break;

    case ANAGLYPH_RB_GRAY:

    case ANAGLYPH_RG_GRAY:

    case ANAGLYPH_RC_GRAY:

    case ANAGLYPH_RC_HALF:

    case ANAGLYPH_RC_COLOR:

    case ANAGLYPH_RC_DUBOIS:

    case ANAGLYPH_GM_GRAY:

    case ANAGLYPH_GM_HALF:

    case ANAGLYPH_GM_COLOR:

    case ANAGLYPH_GM_DUBOIS:

    case ANAGLYPH_YB_GRAY:

    case ANAGLYPH_YB_HALF:

    case ANAGLYPH_YB_COLOR:

    case ANAGLYPH_YB_DUBOIS: {

        ThreadData td;



        td.ileft = ileft; td.iright = iright; td.out = out;

        ctx->internal->execute(ctx, filter_slice, &td, NULL,

                               FFMIN(s->out.height, ctx->graph->nb_threads));

        break;

    }

    case CHECKERBOARD_RL:

    case CHECKERBOARD_LR:

        for (i = 0; i < s->nb_planes; i++) {

            int x, y;



            for (y = 0; y < s->pheight[i]; y++) {

                uint8_t *dst = out->data[i] + out->linesize[i] * y;

                uint8_t *left = ileft->data[i] + ileft->linesize[i] * y + s->in_off_left[i];

                uint8_t *right = iright->data[i] + iright->linesize[i] * y + s->in_off_right[i];

                int p, b;



                if (s->out.format == CHECKERBOARD_RL)

                    FFSWAP(uint8_t*, left, right);

                switch (s->pixstep[i]) {

                case 1:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=2, p++, b++) {

                        dst[x  ] = (b&1) == (y&1) ? left[p] : right[p];

                        dst[x+1] = (b&1) != (y&1) ? left[p] : right[p];

                    }

                    break;

                case 2:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=4, p+=2, b++) {

                        AV_WN16(&dst[x  ], (b&1) == (y&1) ? AV_RN16(&left[p]) : AV_RN16(&right[p]));

                        AV_WN16(&dst[x+2], (b&1) != (y&1) ? AV_RN16(&left[p]) : AV_RN16(&right[p]));

                    }

                    break;

                case 3:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=6, p+=3, b++) {

                        AV_WB24(&dst[x  ], (b&1) == (y&1) ? AV_RB24(&left[p]) : AV_RB24(&right[p]));

                        AV_WB24(&dst[x+3], (b&1) != (y&1) ? AV_RB24(&left[p]) : AV_RB24(&right[p]));

                    }

                    break;

                case 4:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=8, p+=4, b++) {

                        AV_WN32(&dst[x  ], (b&1) == (y&1) ? AV_RN32(&left[p]) : AV_RN32(&right[p]));

                        AV_WN32(&dst[x+4], (b&1) != (y&1) ? AV_RN32(&left[p]) : AV_RN32(&right[p]));

                    }

                    break;

                case 6:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=12, p+=6, b++) {

                        AV_WB48(&dst[x  ], (b&1) == (y&1) ? AV_RB48(&left[p]) : AV_RB48(&right[p]));

                        AV_WB48(&dst[x+6], (b&1) != (y&1) ? AV_RB48(&left[p]) : AV_RB48(&right[p]));

                    }

                    break;

                case 8:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=16, p+=8, b++) {

                        AV_WN64(&dst[x  ], (b&1) == (y&1) ? AV_RN64(&left[p]) : AV_RN64(&right[p]));

                        AV_WN64(&dst[x+8], (b&1) != (y&1) ? AV_RN64(&left[p]) : AV_RN64(&right[p]));

                    }

                    break;

                }

            }

        }

        break;

    case INTERLEAVE_COLS_LR:

    case INTERLEAVE_COLS_RL:

        for (i = 0; i < s->nb_planes; i++) {

            int x, y;



            for (y = 0; y < s->pheight[i]; y++) {

                uint8_t *dst = out->data[i] + out->linesize[i] * y;

                uint8_t *left = ileft->data[i] + ileft->linesize[i] * y * s->in.row_step + s->in_off_left[i];

                uint8_t *right = iright->data[i] + iright->linesize[i] * y * s->in.row_step + s->in_off_right[i];

                int p, b;



                if (s->out.format == INTERLEAVE_COLS_LR)

                    FFSWAP(uint8_t*, left, right);



                switch (s->pixstep[i]) {

                case 1:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=2, p++, b++) {

                        dst[x  ] =   b&1  ? left[p] : right[p];

                        dst[x+1] = !(b&1) ? left[p] : right[p];

                    }

                    break;

                case 2:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=4, p+=2, b++) {

                        AV_WN16(&dst[x  ],   b&1  ? AV_RN16(&left[p]) : AV_RN16(&right[p]));

                        AV_WN16(&dst[x+2], !(b&1) ? AV_RN16(&left[p]) : AV_RN16(&right[p]));

                    }

                    break;

                case 3:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=6, p+=3, b++) {

                        AV_WB24(&dst[x  ],   b&1  ? AV_RB24(&left[p]) : AV_RB24(&right[p]));

                        AV_WB24(&dst[x+3], !(b&1) ? AV_RB24(&left[p]) : AV_RB24(&right[p]));

                    }

                    break;

                case 4:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=8, p+=4, b++) {

                        AV_WN32(&dst[x  ],   b&1  ? AV_RN32(&left[p]) : AV_RN32(&right[p]));

                        AV_WN32(&dst[x+4], !(b&1) ? AV_RN32(&left[p]) : AV_RN32(&right[p]));

                    }

                    break;

                case 6:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=12, p+=6, b++) {

                        AV_WB48(&dst[x  ],   b&1  ? AV_RB48(&left[p]) : AV_RB48(&right[p]));

                        AV_WB48(&dst[x+6], !(b&1) ? AV_RB48(&left[p]) : AV_RB48(&right[p]));

                    }

                    break;

                case 8:

                    for (x = 0, b = 0, p = 0; x < s->linesize[i] * 2; x+=16, p+=8, b++) {

                        AV_WN64(&dst[x  ],   b&1 ?  AV_RN64(&left[p]) : AV_RN64(&right[p]));

                        AV_WN64(&dst[x+8], !(b&1) ? AV_RN64(&left[p]) : AV_RN64(&right[p]));

                    }

                    break;

                }

            }

        }

        break;

    default:

        av_assert0(0);

    }



    av_frame_free(&inpicref);

    av_frame_free(&s->prev);

    if (oright != oleft) {

        if (s->out.format == ALTERNATING_LR)

            FFSWAP(AVFrame *, oleft, oright);

        oright->pts = outlink->frame_count * s->ts_unit;

        ff_filter_frame(outlink, oright);

        out = oleft;

        oleft->pts = outlink->frame_count * s->ts_unit;

    } else if (s->in.format == ALTERNATING_LR ||

               s->in.format == ALTERNATING_RL) {

        out->pts = outlink->frame_count * s->ts_unit;

    }

    return ff_filter_frame(outlink, out);

}
