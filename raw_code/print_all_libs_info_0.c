static void print_all_libs_info(int flags, int level)

{

    PRINT_LIB_INFO(avutil,   AVUTIL,   flags, level);

    PRINT_LIB_INFO(avcodec,  AVCODEC,  flags, level);

    PRINT_LIB_INFO(avformat, AVFORMAT, flags, level);

    PRINT_LIB_INFO(avdevice, AVDEVICE, flags, level);

    PRINT_LIB_INFO(avfilter, AVFILTER, flags, level);

    PRINT_LIB_INFO(avresample, AVRESAMPLE, flags, level);

    PRINT_LIB_INFO(swscale,  SWSCALE,  flags, level);

    PRINT_LIB_INFO(swresample,SWRESAMPLE,  flags, level);

#if CONFIG_POSTPROC

    PRINT_LIB_INFO(postproc, POSTPROC, flags, level);

#endif

}
