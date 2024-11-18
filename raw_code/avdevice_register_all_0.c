void avdevice_register_all(void)

{

    static int inited;



    if (inited)

        return;

    inited = 1;



    /* devices */

    REGISTER_MUXDEMUX (AUDIO_BEOS, audio_beos);

    REGISTER_DEMUXER  (BKTR, bktr);

    REGISTER_DEMUXER  (DV1394, dv1394);

    REGISTER_MUXDEMUX (OSS, oss);

    REGISTER_DEMUXER  (V4L2, v4l2);

    REGISTER_DEMUXER  (V4L, v4l);

    REGISTER_DEMUXER  (X11_GRAB_DEVICE, x11_grab_device);



    /* external libraries */

    REGISTER_DEMUXER  (LIBDC1394, libdc1394);

}
