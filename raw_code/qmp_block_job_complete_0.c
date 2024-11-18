void qmp_block_job_complete(const char *device, Error **errp)

{

    BlockJob *job = find_block_job(device);



    if (!job) {

        error_set(errp, QERR_BLOCK_JOB_NOT_ACTIVE, device);

        return;

    }



    trace_qmp_block_job_complete(job);

    block_job_complete(job, errp);

}
