int ff_unlock_avcodec(const AVCodec *codec)

{

    if (codec->caps_internal & FF_CODEC_CAP_INIT_THREADSAFE || !codec->init)

        return 0;



    av_assert0(ff_avcodec_locked);

    ff_avcodec_locked = 0;

    atomic_fetch_add(&entangled_thread_counter, -1);

    if (lockmgr_cb) {

        if ((*lockmgr_cb)(&codec_mutex, AV_LOCK_RELEASE))

            return -1;

    }



    return 0;

}
