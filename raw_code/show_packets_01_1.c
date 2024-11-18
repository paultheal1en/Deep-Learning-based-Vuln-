static void show_packets(AVFormatContext *fmt_ctx)

{

    AVPacket pkt;



    av_init_packet(&pkt);

    probe_array_header("packets", 0);

    while (!av_read_frame(fmt_ctx, &pkt))

        show_packet(fmt_ctx, &pkt);

    probe_array_footer("packets", 0);

}
