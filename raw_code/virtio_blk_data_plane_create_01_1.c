void virtio_blk_data_plane_create(VirtIODevice *vdev, VirtIOBlkConf *conf,

                                  VirtIOBlockDataPlane **dataplane,

                                  Error **errp)

{

    VirtIOBlockDataPlane *s;

    Error *local_err = NULL;

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(vdev)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);



    *dataplane = NULL;



    if (!conf->data_plane && !conf->iothread) {

        return;

    }



    /* Don't try if transport does not support notifiers. */

    if (!k->set_guest_notifiers || !k->set_host_notifier) {

        error_setg(errp,

                   "device is incompatible with x-data-plane "

                   "(transport does not support notifiers)");

        return;

    }



    /* If dataplane is (re-)enabled while the guest is running there could be

     * block jobs that can conflict.

     */

    if (blk_op_is_blocked(conf->conf.blk, BLOCK_OP_TYPE_DATAPLANE,

                          &local_err)) {

        error_setg(errp, "cannot start dataplane thread: %s",

                   error_get_pretty(local_err));

        error_free(local_err);

        return;

    }



    s = g_new0(VirtIOBlockDataPlane, 1);

    s->vdev = vdev;

    s->conf = conf;



    if (conf->iothread) {

        s->iothread = conf->iothread;

        object_ref(OBJECT(s->iothread));

    } else {

        /* Create per-device IOThread if none specified.  This is for

         * x-data-plane option compatibility.  If x-data-plane is removed we

         * can drop this.

         */

        object_initialize(&s->internal_iothread_obj,

                          sizeof(s->internal_iothread_obj),

                          TYPE_IOTHREAD);

        user_creatable_complete(OBJECT(&s->internal_iothread_obj), &error_abort);

        s->iothread = &s->internal_iothread_obj;

    }

    s->ctx = iothread_get_aio_context(s->iothread);

    s->bh = aio_bh_new(s->ctx, notify_guest_bh, s);



    error_setg(&s->blocker, "block device is in use by data plane");

    blk_op_block_all(conf->conf.blk, s->blocker);

    blk_op_unblock(conf->conf.blk, BLOCK_OP_TYPE_RESIZE, s->blocker);

    blk_op_unblock(conf->conf.blk, BLOCK_OP_TYPE_DRIVE_DEL, s->blocker);

    blk_op_unblock(conf->conf.blk, BLOCK_OP_TYPE_BACKUP_SOURCE, s->blocker);

    blk_op_unblock(conf->conf.blk, BLOCK_OP_TYPE_COMMIT, s->blocker);



    blk_op_unblock(conf->conf.blk, BLOCK_OP_TYPE_MIRROR, s->blocker);

    blk_op_unblock(conf->conf.blk, BLOCK_OP_TYPE_STREAM, s->blocker);

    blk_op_unblock(conf->conf.blk, BLOCK_OP_TYPE_REPLACE, s->blocker);



    *dataplane = s;

}