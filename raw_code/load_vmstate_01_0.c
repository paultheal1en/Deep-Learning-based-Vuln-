int load_vmstate(const char *name)

{

    BlockDriverState *bs, *bs_vm_state;

    QEMUSnapshotInfo sn;

    QEMUFile *f;

    int ret;



    if (!bdrv_all_can_snapshot(&bs)) {

        error_report("Device '%s' is writable but does not support snapshots.",

                     bdrv_get_device_name(bs));

        return -ENOTSUP;

    }



    bs_vm_state = find_vmstate_bs();

    if (!bs_vm_state) {

        error_report("No block device supports snapshots");

        return -ENOTSUP;

    }



    /* Don't even try to load empty VM states */

    ret = bdrv_snapshot_find(bs_vm_state, &sn, name);

    if (ret < 0) {

        return ret;

    } else if (sn.vm_state_size == 0) {

        error_report("This is a disk-only snapshot. Revert to it offline "

            "using qemu-img.");

        return -EINVAL;

    }



    /* Verify if there is any device that doesn't support snapshots and is

    writable and check if the requested snapshot is available too. */

    bs = NULL;

    while ((bs = bdrv_next(bs))) {

        if (!bdrv_can_snapshot(bs)) {

            continue;

        }



        ret = bdrv_snapshot_find(bs, &sn, name);

        if (ret < 0) {

            error_report("Device '%s' does not have the requested snapshot '%s'",

                           bdrv_get_device_name(bs), name);

            return ret;

        }

    }



    /* Flush all IO requests so they don't interfere with the new state.  */

    bdrv_drain_all();



    ret = bdrv_all_goto_snapshot(name, &bs);

    if (ret < 0) {

        error_report("Error %d while activating snapshot '%s' on '%s'",

                     ret, name, bdrv_get_device_name(bs));

        return ret;

    }



    /* restore the VM state */

    f = qemu_fopen_bdrv(bs_vm_state, 0);

    if (!f) {

        error_report("Could not open VM state file");

        return -EINVAL;

    }



    qemu_system_reset(VMRESET_SILENT);

    migration_incoming_state_new(f);

    ret = qemu_loadvm_state(f);



    qemu_fclose(f);

    migration_incoming_state_destroy();

    if (ret < 0) {

        error_report("Error %d while loading VM state", ret);

        return ret;

    }



    return 0;

}
