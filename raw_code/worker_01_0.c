static void* attribute_align_arg worker(void *v)

{

    AVCodecContext *avctx = v;

    SliceThreadContext *c = avctx->internal->thread_ctx;

    unsigned last_execute = 0;

    int our_job = c->job_count;

    int thread_count = avctx->thread_count;

    int self_id;



    pthread_mutex_lock(&c->current_job_lock);

    self_id = c->current_job++;

    for (;;){

        while (our_job >= c->job_count) {

            if (c->current_job == thread_count + c->job_count)

                pthread_cond_signal(&c->last_job_cond);



            while (last_execute == c->current_execute && !c->done)

                pthread_cond_wait(&c->current_job_cond, &c->current_job_lock);

            last_execute = c->current_execute;

            our_job = self_id;



            if (c->done) {

                pthread_mutex_unlock(&c->current_job_lock);

                return NULL;

            }

        }

        pthread_mutex_unlock(&c->current_job_lock);



        c->rets[our_job%c->rets_count] = c->func ? c->func(avctx, (char*)c->args + our_job*c->job_size):

                                                   c->func2(avctx, c->args, our_job, self_id);



        pthread_mutex_lock(&c->current_job_lock);

        our_job = c->current_job++;

    }

}
