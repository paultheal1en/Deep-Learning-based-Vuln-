static void block_job_completed_single(BlockJob *job)

{

    if (!job->ret) {

        if (job->driver->commit) {

            job->driver->commit(job);


    } else {

        if (job->driver->abort) {

            job->driver->abort(job);








    if (job->cb) {

        job->cb(job->opaque, job->ret);


    if (block_job_is_cancelled(job)) {

        block_job_event_cancelled(job);

    } else {

        const char *msg = NULL;

        if (job->ret < 0) {

            msg = strerror(-job->ret);


        block_job_event_completed(job, msg);




    if (job->txn) {

        QLIST_REMOVE(job, txn_list);

        block_job_txn_unref(job->txn);


    block_job_unref(job);
