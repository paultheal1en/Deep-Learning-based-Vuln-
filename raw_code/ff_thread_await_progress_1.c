void ff_thread_await_progress(ThreadFrame *f, int n, int field)

{

    PerThreadContext *p;

    atomic_int *progress = f->progress ? (atomic_int*)f->progress->data : NULL;



    if (!progress ||

        atomic_load_explicit(&progress[field], memory_order_acquire) >= n)

        return;



    p = f->owner[field]->internal->thread_ctx;



    pthread_mutex_lock(&p->progress_mutex);

    if (f->owner[field]->debug&FF_DEBUG_THREADS)

        av_log(f->owner[field], AV_LOG_DEBUG,

               "thread awaiting %d field %d from %p\n", n, field, progress);

    while (atomic_load_explicit(&progress[field], memory_order_relaxed) < n)

        pthread_cond_wait(&p->progress_cond, &p->progress_mutex);

    pthread_mutex_unlock(&p->progress_mutex);

}
