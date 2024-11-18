void block_job_resume(BlockJob *job)

{

    job->paused = false;

    block_job_iostatus_reset(job);

    if (job->co && !job->busy) {

        qemu_coroutine_enter(job->co, NULL);

    }

}
