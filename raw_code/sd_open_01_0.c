static int sd_open(BlockDriverState *bs, QDict *options, int flags,

                   Error **errp)

{

    int ret, fd;

    uint32_t vid = 0;

    BDRVSheepdogState *s = bs->opaque;

    char vdi[SD_MAX_VDI_LEN], tag[SD_MAX_VDI_TAG_LEN];

    uint32_t snapid;

    char *buf = NULL;

    QemuOpts *opts;

    Error *local_err = NULL;

    const char *filename;



    s->bs = bs;

    s->aio_context = bdrv_get_aio_context(bs);



    opts = qemu_opts_create(&runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto out;

    }



    filename = qemu_opt_get(opts, "filename");



    QLIST_INIT(&s->inflight_aio_head);

    QLIST_INIT(&s->failed_aio_head);

    QLIST_INIT(&s->inflight_aiocb_head);

    s->fd = -1;



    memset(vdi, 0, sizeof(vdi));

    memset(tag, 0, sizeof(tag));



    if (strstr(filename, "://")) {

        ret = sd_parse_uri(s, filename, vdi, &snapid, tag);

    } else {

        ret = parse_vdiname(s, filename, vdi, &snapid, tag);

    }

    if (ret < 0) {

        error_setg(errp, "Can't parse filename");

        goto out;

    }

    s->fd = get_sheep_fd(s, errp);

    if (s->fd < 0) {

        ret = s->fd;

        goto out;

    }



    ret = find_vdi_name(s, vdi, snapid, tag, &vid, true, errp);

    if (ret) {

        goto out;

    }



    /*

     * QEMU block layer emulates writethrough cache as 'writeback + flush', so

     * we always set SD_FLAG_CMD_CACHE (writeback cache) as default.

     */

    s->cache_flags = SD_FLAG_CMD_CACHE;

    if (flags & BDRV_O_NOCACHE) {

        s->cache_flags = SD_FLAG_CMD_DIRECT;

    }

    s->discard_supported = true;



    if (snapid || tag[0] != '\0') {

        DPRINTF("%" PRIx32 " snapshot inode was open.\n", vid);

        s->is_snapshot = true;

    }



    fd = connect_to_sdog(s, errp);

    if (fd < 0) {

        ret = fd;

        goto out;

    }



    buf = g_malloc(SD_INODE_SIZE);

    ret = read_object(fd, s->aio_context, buf, vid_to_vdi_oid(vid),

                      0, SD_INODE_SIZE, 0, s->cache_flags);



    closesocket(fd);



    if (ret) {

        error_setg(errp, "Can't read snapshot inode");

        goto out;

    }



    memcpy(&s->inode, buf, sizeof(s->inode));

    s->min_dirty_data_idx = UINT32_MAX;

    s->max_dirty_data_idx = 0;



    bs->total_sectors = s->inode.vdi_size / BDRV_SECTOR_SIZE;

    pstrcpy(s->name, sizeof(s->name), vdi);

    qemu_co_mutex_init(&s->lock);

    qemu_co_queue_init(&s->overwrapping_queue);

    qemu_opts_del(opts);

    g_free(buf);

    return 0;

out:

    aio_set_fd_handler(bdrv_get_aio_context(bs), s->fd, NULL, NULL, NULL);

    if (s->fd >= 0) {

        closesocket(s->fd);

    }

    qemu_opts_del(opts);

    g_free(buf);

    return ret;

}
