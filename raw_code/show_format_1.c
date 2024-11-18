static void show_format(WriterContext *w, AVFormatContext *fmt_ctx)

{

    char val_str[128];

    int64_t size = fmt_ctx->pb ? avio_size(fmt_ctx->pb) : -1;



    print_section_header("format");

    print_str("filename",         fmt_ctx->filename);

    print_int("nb_streams",       fmt_ctx->nb_streams);

    print_str("format_name",      fmt_ctx->iformat->name);

    print_str("format_long_name", fmt_ctx->iformat->long_name);

    print_time("start_time",      fmt_ctx->start_time, &AV_TIME_BASE_Q);

    print_time("duration",        fmt_ctx->duration,   &AV_TIME_BASE_Q);

    if (size >= 0) print_val    ("size", size, unit_byte_str);

    else           print_str_opt("size", "N/A");

    if (fmt_ctx->bit_rate > 0) print_val    ("bit_rate", fmt_ctx->bit_rate, unit_bit_per_second_str);

    else                       print_str_opt("bit_rate", "N/A");

    show_tags(fmt_ctx->metadata);

    print_section_footer("format");

    fflush(stdout);

}
