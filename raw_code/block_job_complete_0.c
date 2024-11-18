void block_job_complete(BlockJob *job, Error **errp)

{

    if (job->paused || job->cancelled || !job->driver->complete) {

        error_set(errp, QERR_BLOCK_JOB_NOT_READY,

                  bdrv_get_device_name(job->bs));

        return;

    }



    job->driver->complete(job, errp);

}
