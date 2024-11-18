void qmp_blockdev_add(BlockdevOptions *options, Error **errp)

{

    BlockDriverState *bs;

    QObject *obj;

    Visitor *v = qmp_output_visitor_new(&obj);

    QDict *qdict;

    Error *local_err = NULL;



    visit_type_BlockdevOptions(v, NULL, &options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto fail;

    }



    visit_complete(v, &obj);

    qdict = qobject_to_qdict(obj);



    qdict_flatten(qdict);



    if (!qdict_get_try_str(qdict, "node-name")) {

        error_setg(errp, "'node-name' must be specified for the root node");

        goto fail;

    }



    bs = bds_tree_init(qdict, errp);

    if (!bs) {

        goto fail;

    }



    QTAILQ_INSERT_TAIL(&monitor_bdrv_states, bs, monitor_list);



    if (bs && bdrv_key_required(bs)) {

        QTAILQ_REMOVE(&monitor_bdrv_states, bs, monitor_list);

        bdrv_unref(bs);

        error_setg(errp, "blockdev-add doesn't support encrypted devices");

        goto fail;

    }



fail:

    visit_free(v);

}
