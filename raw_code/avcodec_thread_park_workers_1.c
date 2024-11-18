static av_always_inline void avcodec_thread_park_workers(ThreadContext *c, int thread_count)

{

    pthread_cond_wait(&c->last_job_cond, &c->current_job_lock);

    pthread_mutex_unlock(&c->current_job_lock);

}
