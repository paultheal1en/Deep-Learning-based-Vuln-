static inline void rgb2rgb_init_c(void)

{

    rgb15to16          = rgb15to16_c;

    rgb15tobgr24       = rgb15tobgr24_c;

    rgb15to32          = rgb15to32_c;

    rgb16tobgr24       = rgb16tobgr24_c;

    rgb16to32          = rgb16to32_c;

    rgb16to15          = rgb16to15_c;

    rgb24tobgr16       = rgb24tobgr16_c;

    rgb24tobgr15       = rgb24tobgr15_c;

    rgb24tobgr32       = rgb24tobgr32_c;

    rgb32to16          = rgb32to16_c;

    rgb32to15          = rgb32to15_c;

    rgb32tobgr24       = rgb32tobgr24_c;

    rgb24to15          = rgb24to15_c;

    rgb24to16          = rgb24to16_c;

    rgb24tobgr24       = rgb24tobgr24_c;

    shuffle_bytes_2103 = shuffle_bytes_2103_c;

    rgb32tobgr16       = rgb32tobgr16_c;

    rgb32tobgr15       = rgb32tobgr15_c;

    yv12toyuy2         = yv12toyuy2_c;

    yv12touyvy         = yv12touyvy_c;

    yuv422ptoyuy2      = yuv422ptoyuy2_c;

    yuv422ptouyvy      = yuv422ptouyvy_c;

    yuy2toyv12         = yuy2toyv12_c;

    planar2x           = planar2x_c;

    rgb24toyv12        = rgb24toyv12_c;

    interleaveBytes    = interleaveBytes_c;

    vu9_to_vu12        = vu9_to_vu12_c;

    yvu9_to_yuy2       = yvu9_to_yuy2_c;



    uyvytoyuv420       = uyvytoyuv420_c;

    uyvytoyuv422       = uyvytoyuv422_c;

    yuyvtoyuv420       = yuyvtoyuv420_c;

    yuyvtoyuv422       = yuyvtoyuv422_c;

}
