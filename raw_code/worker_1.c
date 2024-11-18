static void* attribute_align_arg worker(void *v)

{

    ThreadContext *c = v;

    int our_job      = c->nb_jobs;

    int nb_threads   = c->nb_threads;

    unsigned int last_execute = 0;

    int self_id;



    pthread_mutex_lock(&c->current_job_lock);

    self_id = c->current_job++;

    for (;;) {

        while (our_job >= c->nb_jobs) {

            if (c->current_job == nb_threads + c->nb_jobs)

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



        c->rets[our_job % c->nb_rets] = c->func(c->ctx, c->arg, our_job, c->nb_jobs);



        pthread_mutex_lock(&c->current_job_lock);

        our_job = c->current_job++;

    }

}
