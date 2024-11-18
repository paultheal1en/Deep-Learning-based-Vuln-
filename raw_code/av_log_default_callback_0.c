void av_log_default_callback(void* ptr, int level, const char* fmt, va_list vl)

{

    static int print_prefix = 1;

    static int count;

    static char prev[LINE_SZ];

    AVBPrint part[4];

    char line[LINE_SZ];

    static int is_atty;

    int type[2];

    unsigned tint = 0;



    if (level >= 0) {

        tint = level & 0xff00;

        level &= 0xff;

    }



    if (level > av_log_level)

        return;

#if HAVE_PTHREADS

    pthread_mutex_lock(&mutex);

#endif



    format_line(ptr, level, fmt, vl, part, &print_prefix, type);

    snprintf(line, sizeof(line), "%s%s%s%s", part[0].str, part[1].str, part[2].str, part[3].str);



#if HAVE_ISATTY

    if (!is_atty)

        is_atty = isatty(2) ? 1 : -1;

#endif



    if (print_prefix && (flags & AV_LOG_SKIP_REPEATED) && !strcmp(line, prev) &&

        *line && line[strlen(line) - 1] != '\r'){

        count++;

        if (is_atty == 1)

            fprintf(stderr, "    Last message repeated %d times\r", count);

        goto end;

    }

    if (count > 0) {

        fprintf(stderr, "    Last message repeated %d times\n", count);

        count = 0;

    }

    strcpy(prev, line);

    sanitize(part[0].str);

    colored_fputs(type[0], 0, part[0].str);

    sanitize(part[1].str);

    colored_fputs(type[1], 0, part[1].str);

    sanitize(part[2].str);

    colored_fputs(av_clip(level >> 3, 0, NB_LEVELS - 1), tint >> 8, part[2].str);

    sanitize(part[3].str);

    colored_fputs(av_clip(level >> 3, 0, NB_LEVELS - 1), tint >> 8, part[3].str);



#if CONFIG_VALGRIND_BACKTRACE

    if (level <= BACKTRACE_LOGLEVEL)

        VALGRIND_PRINTF_BACKTRACE("%s", "");

#endif

end:

    av_bprint_finalize(part+3, NULL);

#if HAVE_PTHREADS

    pthread_mutex_unlock(&mutex);

#endif

}
