static int quorum_open(BlockDriverState *bs, QDict *options, int flags,

                       Error **errp)

{

    BDRVQuorumState *s = bs->opaque;

    Error *local_err = NULL;

    QemuOpts *opts;

    bool *opened;

    QDict *sub = NULL;

    QList *list = NULL;

    const QListEntry *lentry;

    int i;

    int ret = 0;



    qdict_flatten(options);

    qdict_extract_subqdict(options, &sub, "children.");

    qdict_array_split(sub, &list);



    if (qdict_size(sub)) {

        error_setg(&local_err, "Invalid option children.%s",

                   qdict_first(sub)->key);

        ret = -EINVAL;

        goto exit;

    }



    /* count how many different children are present */

    s->num_children = qlist_size(list);

    if (s->num_children < 2) {

        error_setg(&local_err,

                   "Number of provided children must be greater than 1");

        ret = -EINVAL;

        goto exit;

    }



    opts = qemu_opts_create(&quorum_runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        ret = -EINVAL;

        goto exit;

    }



    s->threshold = qemu_opt_get_number(opts, QUORUM_OPT_VOTE_THRESHOLD, 0);



    /* and validate it against s->num_children */

    ret = quorum_valid_threshold(s->threshold, s->num_children, &local_err);

    if (ret < 0) {

        goto exit;

    }



    /* is the driver in blkverify mode */

    if (qemu_opt_get_bool(opts, QUORUM_OPT_BLKVERIFY, false) &&

        s->num_children == 2 && s->threshold == 2) {

        s->is_blkverify = true;

    } else if (qemu_opt_get_bool(opts, QUORUM_OPT_BLKVERIFY, false)) {

        fprintf(stderr, "blkverify mode is set by setting blkverify=on "

                "and using two files with vote_threshold=2\n");

    }



    s->rewrite_corrupted = qemu_opt_get_bool(opts, QUORUM_OPT_REWRITE, false);

    if (s->rewrite_corrupted && s->is_blkverify) {

        error_setg(&local_err,

                   "rewrite-corrupted=on cannot be used with blkverify=on");

        ret = -EINVAL;

        goto exit;

    }



    /* allocate the children BlockDriverState array */

    s->bs = g_new0(BlockDriverState *, s->num_children);

    opened = g_new0(bool, s->num_children);



    for (i = 0, lentry = qlist_first(list); lentry;

         lentry = qlist_next(lentry), i++) {

        QDict *d;

        QString *string;



        switch (qobject_type(lentry->value))

        {

            /* List of options */

            case QTYPE_QDICT:

                d = qobject_to_qdict(lentry->value);

                QINCREF(d);

                ret = bdrv_open(&s->bs[i], NULL, NULL, d, flags, NULL,

                                &local_err);

                break;



            /* QMP reference */

            case QTYPE_QSTRING:

                string = qobject_to_qstring(lentry->value);

                ret = bdrv_open(&s->bs[i], NULL, qstring_get_str(string), NULL,

                                flags, NULL, &local_err);

                break;



            default:

                error_setg(&local_err, "Specification of child block device %i "

                           "is invalid", i);

                ret = -EINVAL;

        }



        if (ret < 0) {

            goto close_exit;

        }

        opened[i] = true;

    }



    g_free(opened);

    goto exit;



close_exit:

    /* cleanup on error */

    for (i = 0; i < s->num_children; i++) {

        if (!opened[i]) {

            continue;

        }

        bdrv_unref(s->bs[i]);

    }

    g_free(s->bs);

    g_free(opened);

exit:

    /* propagate error */

    if (local_err) {

        error_propagate(errp, local_err);

    }

    QDECREF(list);

    QDECREF(sub);

    return ret;

}
