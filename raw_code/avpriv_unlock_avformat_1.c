int avpriv_unlock_avformat(void)

{

    if (lockmgr_cb) {

        if ((*lockmgr_cb)(&avformat_mutex, AV_LOCK_RELEASE))

            return -1;

    }

    return 0;

}
