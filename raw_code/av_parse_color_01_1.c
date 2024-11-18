int av_parse_color(uint8_t *rgba_color, const char *color_string, int slen,

                   void *log_ctx)

{

    char *tail, color_string2[128];

    const ColorEntry *entry;

    int len, hex_offset = 0;



    if (color_string[0] == '#') {

        hex_offset = 1;

    } else if (!strncmp(color_string, "0x", 2))

        hex_offset = 2;



    if (slen < 0)

        slen = strlen(color_string);

    av_strlcpy(color_string2, color_string + hex_offset,

               FFMIN(slen-hex_offset+1, sizeof(color_string2)));

    if ((tail = strchr(color_string2, ALPHA_SEP)))

        *tail++ = 0;

    len = strlen(color_string2);

    rgba_color[3] = 255;



    if (!av_strcasecmp(color_string2, "random") || !av_strcasecmp(color_string2, "bikeshed")) {

        int rgba = av_get_random_seed();

        rgba_color[0] = rgba >> 24;

        rgba_color[1] = rgba >> 16;

        rgba_color[2] = rgba >> 8;

        rgba_color[3] = rgba;

    } else if (hex_offset ||

               strspn(color_string2, "0123456789ABCDEFabcdef") == len) {

        char *tail;

        unsigned int rgba = strtoul(color_string2, &tail, 16);



        if (*tail || (len != 6 && len != 8)) {

            av_log(log_ctx, AV_LOG_ERROR, "Invalid 0xRRGGBB[AA] color string: '%s'\n", color_string2);

            return AVERROR(EINVAL);

        }

        if (len == 8) {

            rgba_color[3] = rgba;

            rgba >>= 8;

        }

        rgba_color[0] = rgba >> 16;

        rgba_color[1] = rgba >> 8;

        rgba_color[2] = rgba;

    } else {

        entry = bsearch(color_string2,

                        color_table,

                        FF_ARRAY_ELEMS(color_table),

                        sizeof(ColorEntry),

                        color_table_compare);

        if (!entry) {

            av_log(log_ctx, AV_LOG_ERROR, "Cannot find color '%s'\n", color_string2);

            return AVERROR(EINVAL);

        }

        memcpy(rgba_color, entry->rgb_color, 3);

    }



    if (tail) {

        unsigned long int alpha;

        const char *alpha_string = tail;

        if (!strncmp(alpha_string, "0x", 2)) {

            alpha = strtoul(alpha_string, &tail, 16);

        } else {

            alpha = 255 * strtod(alpha_string, &tail);

        }



        if (tail == alpha_string || *tail || alpha > 255) {

            av_log(log_ctx, AV_LOG_ERROR, "Invalid alpha value specifier '%s' in '%s'\n",

                   alpha_string, color_string);

            return AVERROR(EINVAL);

        }

        rgba_color[3] = alpha;

    }



    return 0;

}
