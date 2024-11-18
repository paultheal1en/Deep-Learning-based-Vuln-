static void process_incoming_migration_co(void *opaque)

{

    QEMUFile *f = opaque;

    Error *local_err = NULL;

    int ret;



    migration_incoming_state_new(f);

    migrate_generate_event(MIGRATION_STATUS_ACTIVE);

    ret = qemu_loadvm_state(f);



    qemu_fclose(f);

    free_xbzrle_decoded_buf();

    migration_incoming_state_destroy();



    if (ret < 0) {

        migrate_generate_event(MIGRATION_STATUS_FAILED);

        error_report("load of migration failed: %s", strerror(-ret));

        migrate_decompress_threads_join();

        exit(EXIT_FAILURE);

    }

    migrate_generate_event(MIGRATION_STATUS_COMPLETED);

    qemu_announce_self();



    /* Make sure all file formats flush their mutable metadata */

    bdrv_invalidate_cache_all(&local_err);

    if (local_err) {

        error_report_err(local_err);

        migrate_decompress_threads_join();

        exit(EXIT_FAILURE);

    }



    /* If global state section was not received or we are in running

       state, we need to obey autostart. Any other state is set with

       runstate_set. */



    if (!global_state_received() ||

        global_state_get_runstate() == RUN_STATE_RUNNING) {

        if (autostart) {

            vm_start();

        } else {

            runstate_set(RUN_STATE_PAUSED);

        }

    } else {

        runstate_set(global_state_get_runstate());

    }

    migrate_decompress_threads_join();

}
