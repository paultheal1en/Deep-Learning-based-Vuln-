static void show_format(AVFormatContext *fmt_ctx)

{

    AVDictionaryEntry *tag = NULL;

    char val_str[128];

    int64_t size = fmt_ctx->pb ? avio_size(fmt_ctx->pb) : -1;



    print_format_entry(NULL, "[FORMAT]");

    print_format_entry("filename",         fmt_ctx->filename);

    snprintf(val_str, sizeof(val_str) - 1, "%d", fmt_ctx->nb_streams);

    print_format_entry("nb_streams",       val_str);

    print_format_entry("format_name",      fmt_ctx->iformat->name);

    print_format_entry("format_long_name", fmt_ctx->iformat->long_name);

    print_format_entry("start_time",

                       time_value_string(val_str, sizeof(val_str),

                                         fmt_ctx->start_time, &AV_TIME_BASE_Q));

    print_format_entry("duration",

                       time_value_string(val_str, sizeof(val_str),

                                         fmt_ctx->duration, &AV_TIME_BASE_Q));

    print_format_entry("size",

                       size >= 0 ? value_string(val_str, sizeof(val_str),

                                                size, unit_byte_str)

                                  : "unknown");

    print_format_entry("bit_rate",

                       value_string(val_str, sizeof(val_str),

                                    fmt_ctx->bit_rate, unit_bit_per_second_str));



    while ((tag = av_dict_get(fmt_ctx->metadata, "", tag,

                              AV_DICT_IGNORE_SUFFIX))) {

        snprintf(val_str, sizeof(val_str) - 1, "TAG:%s", tag->key);

        print_format_entry(val_str, tag->value);

    }



    print_format_entry(NULL, "[/FORMAT]");

}
