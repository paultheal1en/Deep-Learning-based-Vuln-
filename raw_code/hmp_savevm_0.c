void hmp_savevm(Monitor *mon, const QDict *qdict)

{

    BlockDriverState *bs, *bs1;

    QEMUSnapshotInfo sn1, *sn = &sn1, old_sn1, *old_sn = &old_sn1;

    int ret;

    QEMUFile *f;

    int saved_vm_running;

    uint64_t vm_state_size;

    qemu_timeval tv;

    struct tm tm;

    const char *name = qdict_get_try_str(qdict, "name");

    Error *local_err = NULL;

    AioContext *aio_context;



    if (!bdrv_all_can_snapshot(&bs)) {

        monitor_printf(mon, "Device '%s' is writable but does not "

                       "support snapshots.\n", bdrv_get_device_name(bs));

        return;

    }



    /* Delete old snapshots of the same name */

    if (name && bdrv_all_delete_snapshot(name, &bs1, &local_err) < 0) {

        error_reportf_err(local_err,

                          "Error while deleting snapshot on device '%s': ",

                          bdrv_get_device_name(bs1));

        return;

    }



    bs = bdrv_all_find_vmstate_bs();

    if (bs == NULL) {

        monitor_printf(mon, "No block device can accept snapshots\n");

        return;

    }

    aio_context = bdrv_get_aio_context(bs);



    saved_vm_running = runstate_is_running();



    ret = global_state_store();

    if (ret) {

        monitor_printf(mon, "Error saving global state\n");

        return;

    }

    vm_stop(RUN_STATE_SAVE_VM);



    aio_context_acquire(aio_context);



    memset(sn, 0, sizeof(*sn));



    /* fill auxiliary fields */

    qemu_gettimeofday(&tv);

    sn->date_sec = tv.tv_sec;

    sn->date_nsec = tv.tv_usec * 1000;

    sn->vm_clock_nsec = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);



    if (name) {

        ret = bdrv_snapshot_find(bs, old_sn, name);

        if (ret >= 0) {

            pstrcpy(sn->name, sizeof(sn->name), old_sn->name);

            pstrcpy(sn->id_str, sizeof(sn->id_str), old_sn->id_str);

        } else {

            pstrcpy(sn->name, sizeof(sn->name), name);

        }

    } else {

        /* cast below needed for OpenBSD where tv_sec is still 'long' */

        localtime_r((const time_t *)&tv.tv_sec, &tm);

        strftime(sn->name, sizeof(sn->name), "vm-%Y%m%d%H%M%S", &tm);

    }



    /* save the VM state */

    f = qemu_fopen_bdrv(bs, 1);

    if (!f) {

        monitor_printf(mon, "Could not open VM state file\n");

        goto the_end;

    }

    ret = qemu_savevm_state(f, &local_err);

    vm_state_size = qemu_ftell(f);

    qemu_fclose(f);

    if (ret < 0) {

        error_report_err(local_err);

        goto the_end;

    }



    ret = bdrv_all_create_snapshot(sn, bs, vm_state_size, &bs);

    if (ret < 0) {

        monitor_printf(mon, "Error while creating snapshot on '%s'\n",

                       bdrv_get_device_name(bs));

    }



 the_end:

    aio_context_release(aio_context);

    if (saved_vm_running) {

        vm_start();

    }

}
