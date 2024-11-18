static void loadvm_postcopy_handle_run_bh(void *opaque)

{

    Error *local_err = NULL;

    HandleRunBhData *data = opaque;



    /* TODO we should move all of this lot into postcopy_ram.c or a shared code

     * in migration.c

     */

    cpu_synchronize_all_post_init();



    qemu_announce_self();



    /* Make sure all file formats flush their mutable metadata.

     * If we get an error here, just don't restart the VM yet. */

    bdrv_invalidate_cache_all(&local_err);

    if (!local_err) {

        blk_resume_after_migration(&local_err);

    }

    if (local_err) {

        error_report_err(local_err);

        local_err = NULL;

        autostart = false;

    }



    trace_loadvm_postcopy_handle_run_cpu_sync();

    cpu_synchronize_all_post_init();



    trace_loadvm_postcopy_handle_run_vmstart();



    if (autostart) {

        /* Hold onto your hats, starting the CPU */

        vm_start();

    } else {

        /* leave it paused and let management decide when to start the CPU */

        runstate_set(RUN_STATE_PAUSED);

    }



    qemu_bh_delete(data->bh);

    g_free(data);

}
