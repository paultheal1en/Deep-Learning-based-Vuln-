void qmp_cont(Error **errp)

{

    Error *local_err = NULL;



    if (runstate_check(RUN_STATE_INMIGRATE)) {

        error_set(errp, QERR_MIGRATION_EXPECTED);

        return;

    } else if (runstate_check(RUN_STATE_INTERNAL_ERROR) ||

               runstate_check(RUN_STATE_SHUTDOWN)) {

        error_set(errp, QERR_RESET_REQUIRED);

        return;

    } else if (runstate_check(RUN_STATE_SUSPENDED)) {

        return;

    }



    bdrv_iterate(iostatus_bdrv_it, NULL);

    bdrv_iterate(encrypted_bdrv_it, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    vm_start();

}
