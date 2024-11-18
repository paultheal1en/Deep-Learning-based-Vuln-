bool st_init(const char *file)

{

    pthread_t thread;

    pthread_attr_t attr;

    sigset_t set, oldset;

    int ret;



    pthread_attr_init(&attr);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);



    sigfillset(&set);

    pthread_sigmask(SIG_SETMASK, &set, &oldset);

    ret = pthread_create(&thread, &attr, writeout_thread, NULL);

    pthread_sigmask(SIG_SETMASK, &oldset, NULL);



    if (ret != 0) {

        return false;

    }



    atexit(st_flush_trace_buffer);

    st_set_trace_file(file);

    return true;

}
