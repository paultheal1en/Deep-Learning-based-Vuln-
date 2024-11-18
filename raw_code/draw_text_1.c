static int draw_text(AVFilterContext *ctx, AVFilterBufferRef *picref,

                     int width, int height)

{

    DrawTextContext *dtext = ctx->priv;

    uint32_t code = 0, prev_code = 0;

    int x = 0, y = 0, i = 0, ret;

    int max_text_line_w = 0, len;

    int box_w, box_h;

    char *text = dtext->text;

    uint8_t *p;

    int y_min = 32000, y_max = -32000;

    int x_min = 32000, x_max = -32000;

    FT_Vector delta;

    Glyph *glyph = NULL, *prev_glyph = NULL;

    Glyph dummy = { 0 };



    time_t now = time(0);

    struct tm ltime;

    uint8_t *buf = dtext->expanded_text;

    int buf_size = dtext->expanded_text_size;



    if(dtext->basetime != AV_NOPTS_VALUE)

        now= picref->pts*av_q2d(ctx->inputs[0]->time_base) + dtext->basetime/1000000;



    if (!buf) {

        buf_size = 2*strlen(dtext->text)+1;

        buf = av_malloc(buf_size);

    }



#if HAVE_LOCALTIME_R

    localtime_r(&now, &ltime);

#else

    if(strchr(dtext->text, '%'))

        ltime= *localtime(&now);

#endif



    do {

        *buf = 1;

        if (strftime(buf, buf_size, dtext->text, &ltime) != 0 || *buf == 0)

            break;

        buf_size *= 2;

    } while ((buf = av_realloc(buf, buf_size)));



    if (dtext->tc_opt_string) {

        char tcbuf[AV_TIMECODE_STR_SIZE];

        av_timecode_make_string(&dtext->tc, tcbuf, dtext->frame_id++);


        buf = av_asprintf("%s%s", dtext->text, tcbuf);

    }



    if (!buf)

        return AVERROR(ENOMEM);

    text = dtext->expanded_text = buf;

    dtext->expanded_text_size = buf_size;

    if ((len = strlen(text)) > dtext->nb_positions) {

        if (!(dtext->positions =

              av_realloc(dtext->positions, len*sizeof(*dtext->positions))))

            return AVERROR(ENOMEM);

        dtext->nb_positions = len;

    }



    x = 0;

    y = 0;



    /* load and cache glyphs */

    for (i = 0, p = text; *p; i++) {

        GET_UTF8(code, *p++, continue;);



        /* get glyph */

        dummy.code = code;

        glyph = av_tree_find(dtext->glyphs, &dummy, glyph_cmp, NULL);

        if (!glyph) {

            load_glyph(ctx, &glyph, code);

        }



        y_min = FFMIN(glyph->bbox.yMin, y_min);

        y_max = FFMAX(glyph->bbox.yMax, y_max);

        x_min = FFMIN(glyph->bbox.xMin, x_min);

        x_max = FFMAX(glyph->bbox.xMax, x_max);

    }

    dtext->max_glyph_h = y_max - y_min;

    dtext->max_glyph_w = x_max - x_min;



    /* compute and save position for each glyph */

    glyph = NULL;

    for (i = 0, p = text; *p; i++) {

        GET_UTF8(code, *p++, continue;);



        /* skip the \n in the sequence \r\n */

        if (prev_code == '\r' && code == '\n')

            continue;



        prev_code = code;

        if (is_newline(code)) {

            max_text_line_w = FFMAX(max_text_line_w, x);

            y += dtext->max_glyph_h;

            x = 0;

            continue;

        }



        /* get glyph */

        prev_glyph = glyph;

        dummy.code = code;

        glyph = av_tree_find(dtext->glyphs, &dummy, glyph_cmp, NULL);



        /* kerning */

        if (dtext->use_kerning && prev_glyph && glyph->code) {

            FT_Get_Kerning(dtext->face, prev_glyph->code, glyph->code,

                           ft_kerning_default, &delta);

            x += delta.x >> 6;

        }



        /* save position */

        dtext->positions[i].x = x + glyph->bitmap_left;

        dtext->positions[i].y = y - glyph->bitmap_top + y_max;

        if (code == '\t') x  = (x / dtext->tabsize + 1)*dtext->tabsize;

        else              x += glyph->advance;

    }



    max_text_line_w = FFMAX(x, max_text_line_w);



    dtext->var_values[VAR_TW] = dtext->var_values[VAR_TEXT_W] = max_text_line_w;

    dtext->var_values[VAR_TH] = dtext->var_values[VAR_TEXT_H] = y + dtext->max_glyph_h;



    dtext->var_values[VAR_MAX_GLYPH_W] = dtext->max_glyph_w;

    dtext->var_values[VAR_MAX_GLYPH_H] = dtext->max_glyph_h;

    dtext->var_values[VAR_MAX_GLYPH_A] = dtext->var_values[VAR_ASCENT ] = y_max;

    dtext->var_values[VAR_MAX_GLYPH_D] = dtext->var_values[VAR_DESCENT] = y_min;



    dtext->var_values[VAR_LINE_H] = dtext->var_values[VAR_LH] = dtext->max_glyph_h;



    dtext->x = dtext->var_values[VAR_X] = av_expr_eval(dtext->x_pexpr, dtext->var_values, &dtext->prng);

    dtext->y = dtext->var_values[VAR_Y] = av_expr_eval(dtext->y_pexpr, dtext->var_values, &dtext->prng);

    dtext->x = dtext->var_values[VAR_X] = av_expr_eval(dtext->x_pexpr, dtext->var_values, &dtext->prng);

    dtext->draw = av_expr_eval(dtext->draw_pexpr, dtext->var_values, &dtext->prng);



    if(!dtext->draw)

        return 0;



    box_w = FFMIN(width - 1 , max_text_line_w);

    box_h = FFMIN(height - 1, y + dtext->max_glyph_h);



    /* draw box */

    if (dtext->draw_box)

        ff_blend_rectangle(&dtext->dc, &dtext->boxcolor,

                           picref->data, picref->linesize, width, height,

                           dtext->x, dtext->y, box_w, box_h);



    if (dtext->shadowx || dtext->shadowy) {

        if ((ret = draw_glyphs(dtext, picref, width, height, dtext->shadowcolor.rgba,

                               &dtext->shadowcolor, dtext->shadowx, dtext->shadowy)) < 0)

            return ret;

    }



    if ((ret = draw_glyphs(dtext, picref, width, height, dtext->fontcolor.rgba,

                           &dtext->fontcolor, 0, 0)) < 0)

        return ret;



    return 0;

}