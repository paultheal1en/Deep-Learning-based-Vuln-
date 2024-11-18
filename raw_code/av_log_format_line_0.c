void av_log_format_line(void *ptr, int level, const char *fmt, va_list vl,

                        char *line, int line_size, int *print_prefix)

{

    AVBPrint part[4];

    format_line(ptr, level, fmt, vl, part, print_prefix, NULL);

    snprintf(line, line_size, "%s%s%s%s", part[0].str, part[1].str, part[2].str, part[3].str);

    av_bprint_finalize(part+3, NULL);

}
