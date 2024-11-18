static int virtio_9p_device_init(VirtIODevice *vdev)

{

    V9fsState *s = VIRTIO_9P(vdev);

    int i, len;

    struct stat stat;

    FsDriverEntry *fse;

    V9fsPath path;



    virtio_init(VIRTIO_DEVICE(s), "virtio-9p", VIRTIO_ID_9P,

                sizeof(struct virtio_9p_config) + MAX_TAG_LEN);



    /* initialize pdu allocator */

    QLIST_INIT(&s->free_list);

    QLIST_INIT(&s->active_list);

    for (i = 0; i < (MAX_REQ - 1); i++) {

        QLIST_INSERT_HEAD(&s->free_list, &s->pdus[i], next);

    }



    s->vq = virtio_add_queue(vdev, MAX_REQ, handle_9p_output);



    fse = get_fsdev_fsentry(s->fsconf.fsdev_id);



    if (!fse) {

        /* We don't have a fsdev identified by fsdev_id */

        fprintf(stderr, "Virtio-9p device couldn't find fsdev with the "

                "id = %s\n",

                s->fsconf.fsdev_id ? s->fsconf.fsdev_id : "NULL");

        goto out;

    }



    if (!s->fsconf.tag) {

        /* we haven't specified a mount_tag */

        fprintf(stderr, "fsdev with id %s needs mount_tag arguments\n",

                s->fsconf.fsdev_id);

        goto out;

    }



    s->ctx.export_flags = fse->export_flags;

    s->ctx.fs_root = g_strdup(fse->path);

    s->ctx.exops.get_st_gen = NULL;

    len = strlen(s->fsconf.tag);

    if (len > MAX_TAG_LEN - 1) {

        fprintf(stderr, "mount tag '%s' (%d bytes) is longer than "

                "maximum (%d bytes)", s->fsconf.tag, len, MAX_TAG_LEN - 1);

        goto out;

    }



    s->tag = g_strdup(s->fsconf.tag);

    s->ctx.uid = -1;



    s->ops = fse->ops;

    s->config_size = sizeof(struct virtio_9p_config) + len;

    s->fid_list = NULL;

    qemu_co_rwlock_init(&s->rename_lock);



    if (s->ops->init(&s->ctx) < 0) {

        fprintf(stderr, "Virtio-9p Failed to initialize fs-driver with id:%s"

                " and export path:%s\n", s->fsconf.fsdev_id, s->ctx.fs_root);

        goto out;

    }

    if (v9fs_init_worker_threads() < 0) {

        fprintf(stderr, "worker thread initialization failed\n");

        goto out;

    }



    /*

     * Check details of export path, We need to use fs driver

     * call back to do that. Since we are in the init path, we don't

     * use co-routines here.

     */

    v9fs_path_init(&path);

    if (s->ops->name_to_path(&s->ctx, NULL, "/", &path) < 0) {

        fprintf(stderr,

                "error in converting name to path %s", strerror(errno));

        goto out;

    }

    if (s->ops->lstat(&s->ctx, &path, &stat)) {

        fprintf(stderr, "share path %s does not exist\n", fse->path);

        goto out;

    } else if (!S_ISDIR(stat.st_mode)) {

        fprintf(stderr, "share path %s is not a directory\n", fse->path);

        goto out;

    }

    v9fs_path_free(&path);



    return 0;

out:

    g_free(s->ctx.fs_root);

    g_free(s->tag);

    virtio_cleanup(vdev);

    v9fs_path_free(&path);



    return -1;



}
