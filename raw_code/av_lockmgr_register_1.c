int av_lockmgr_register(int (*cb)(void **mutex, enum AVLockOp op))

{

    if (lockmgr_cb) {

        // There is no good way to rollback a failure to destroy the

        // mutex, so we ignore failures.

        lockmgr_cb(&codec_mutex,    AV_LOCK_DESTROY);

        lockmgr_cb(&avformat_mutex, AV_LOCK_DESTROY);

        lockmgr_cb     = NULL;

        codec_mutex    = NULL;

        avformat_mutex = NULL;

    }



    if (cb) {

        void *new_codec_mutex    = NULL;

        void *new_avformat_mutex = NULL;

        int err;

        if (err = cb(&new_codec_mutex, AV_LOCK_CREATE)) {

            return err > 0 ? AVERROR_UNKNOWN : err;

        }

        if (err = cb(&new_avformat_mutex, AV_LOCK_CREATE)) {

            // Ignore failures to destroy the newly created mutex.

            cb(&new_codec_mutex, AV_LOCK_DESTROY);

            return err > 0 ? AVERROR_UNKNOWN : err;

        }

        lockmgr_cb     = cb;

        codec_mutex    = new_codec_mutex;

        avformat_mutex = new_avformat_mutex;

    }



    return 0;

}
