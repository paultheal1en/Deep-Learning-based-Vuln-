static int probe_file(const char *filename)

{

    AVFormatContext *fmt_ctx;

    int ret, i;



    if ((ret = open_input_file(&fmt_ctx, filename)))

        return ret;



    if (do_show_packets)

        show_packets(fmt_ctx);



    if (do_show_streams)

        for (i = 0; i < fmt_ctx->nb_streams; i++)

            show_stream(fmt_ctx, i);



    if (do_show_format)

        show_format(fmt_ctx);



    close_input_file(&fmt_ctx);

    return 0;

}
