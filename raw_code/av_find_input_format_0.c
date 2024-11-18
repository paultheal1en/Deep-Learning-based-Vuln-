AVInputFormat *av_find_input_format(const char *short_name)

{

    AVInputFormat *fmt = NULL;

    while ((fmt = av_iformat_next(fmt)))

        if (match_format(short_name, fmt->name))

            return fmt;

    return NULL;

}
