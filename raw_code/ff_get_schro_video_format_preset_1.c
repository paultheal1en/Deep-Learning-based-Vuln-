SchroVideoFormatEnum ff_get_schro_video_format_preset(AVCodecContext *avctx)

{

    unsigned int num_formats = sizeof(ff_schro_video_formats) /

                               sizeof(ff_schro_video_formats[0]);



    unsigned int idx = get_video_format_idx(avctx);



    return (idx < num_formats) ? ff_schro_video_formats[idx] :

                                 SCHRO_VIDEO_FORMAT_CUSTOM;

}
