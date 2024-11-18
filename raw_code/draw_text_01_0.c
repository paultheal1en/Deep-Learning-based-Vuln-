static int draw_text(AVFilterContext *ctx, AVFilterBufferRef *picref,

                     int width, int height)

{

    DrawTextContext *dtext = ctx->priv;

    uint32_t code = 0, prev_code = 0;

    int x = 0, y = 0, i = 0, ret;

    int text_height;

    char *text = dtext->text;

    uint8_t *p;

    int str_w = 0, len;

    int y_min = 32000, y_max = -32000;

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



    x = dtext->x;

    y = dtext->y;



    /* load and cache glyphs */

    for (i = 0, p = text; *p; i++) {

        GET_UTF8(code, *p++, continue;);



        /* get glyph */

        dummy.code = code;

        glyph = av_tree_find(dtext->glyphs, &dummy, glyph_cmp, NULL);

        if (!glyph)

            load_glyph(ctx, &glyph, code);



        y_min = FFMIN(glyph->bbox.yMin, y_min);

        y_max = FFMAX(glyph->bbox.yMax, y_max);

    }

    text_height = y_max - y_min;



    /* compute and save position for each glyph */

    glyph = NULL;

    for (i = 0, p = text; *p; i++) {

        GET_UTF8(code, *p++, continue;);



        /* skip the \n in the sequence \r\n */

        if (prev_code == '\r' && code == '\n')

            continue;



        prev_code = code;

        if (is_newline(code)) {

            str_w = FFMAX(str_w, x - dtext->x);

            y += text_height;

            x = dtext->x;

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



        if (x + glyph->bbox.xMax >= width) {

            str_w = FFMAX(str_w, x - dtext->x);

            y += text_height;

            x = dtext->x;

        }



        /* save position */

        dtext->positions[i].x = x + glyph->bitmap_left;

        dtext->positions[i].y = y - glyph->bitmap_top + y_max;

        if (code == '\t') x  = (x / dtext->tabsize + 1)*dtext->tabsize;

        else              x += glyph->advance;

    }



    str_w = FFMIN(width - dtext->x - 1, FFMAX(str_w, x - dtext->x));

    y     = FFMIN(y + text_height, height - 1);



    /* draw box */

    if (dtext->draw_box)

        drawbox(picref, dtext->x, dtext->y, str_w, y-dtext->y,

                dtext->box_line, dtext->pixel_step, dtext->boxcolor_rgba,

                dtext->hsub, dtext->vsub, dtext->is_packed_rgb, dtext->rgba_map);



    if (dtext->shadowx || dtext->shadowy) {

        if ((ret = draw_glyphs(dtext, picref, width, height, dtext->shadowcolor_rgba,

                               dtext->shadowcolor, dtext->shadowx, dtext->shadowy)) < 0)

            return ret;

    }



    if ((ret = draw_glyphs(dtext, picref, width, height, dtext->fontcolor_rgba,

                           dtext->fontcolor, 0, 0)) < 0)

        return ret;



    return 0;

}
