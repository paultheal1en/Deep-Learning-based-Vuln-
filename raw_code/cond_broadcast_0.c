static void cond_broadcast(pthread_cond_t *cond)

{

    int ret = pthread_cond_broadcast(cond);

    if (ret) die2(ret, "pthread_cond_broadcast");

}
