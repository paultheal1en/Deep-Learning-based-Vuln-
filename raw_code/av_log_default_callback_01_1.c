void av_log_default_callback(void *avcl, int level, const char *fmt, va_list vl)

{

    static int print_prefix = 1;

    static int count;

    static char prev[1024];

    char line[1024];

    static int is_atty;

    AVClass* avc = avcl ? *(AVClass **) avcl : NULL;

    int tint = av_clip(level >> 8, 0, 256);



    level &= 0xff;



    if (level > av_log_level)

        return;

    line[0] = 0;

    if (print_prefix && avc) {

        if (avc->parent_log_context_offset) {

            AVClass** parent = *(AVClass ***) (((uint8_t *) avcl) +

                                   avc->parent_log_context_offset);

            if (parent && *parent) {

                snprintf(line, sizeof(line), "[%s @ %p] ",

                         (*parent)->item_name(parent), parent);

            }

        }

        snprintf(line + strlen(line), sizeof(line) - strlen(line), "[%s @ %p] ",

                 avc->item_name(avcl), avcl);

    }



    vsnprintf(line + strlen(line), sizeof(line) - strlen(line), fmt, vl);



    print_prefix = strlen(line) && line[strlen(line) - 1] == '\n';



#if HAVE_ISATTY

    if (!is_atty)

        is_atty = isatty(2) ? 1 : -1;

#endif



    if (print_prefix && (flags & AV_LOG_SKIP_REPEATED) &&

        !strncmp(line, prev, sizeof line)) {

        count++;

        if (is_atty == 1)

            fprintf(stderr, "    Last message repeated %d times\r", count);

        return;

    }

    if (count > 0) {

        fprintf(stderr, "    Last message repeated %d times\n", count);

        count = 0;

    }

    colored_fputs(av_clip(level >> 3, 0, 6), tint, line);

    av_strlcpy(prev, line, sizeof line);

}
