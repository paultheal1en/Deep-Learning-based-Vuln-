static NFSServer *nfs_config(QDict *options, Error **errp)

{

    NFSServer *server = NULL;

    QDict *addr = NULL;

    QObject *crumpled_addr = NULL;

    Visitor *iv = NULL;

    Error *local_error = NULL;



    qdict_extract_subqdict(options, &addr, "server.");

    if (!qdict_size(addr)) {

        error_setg(errp, "NFS server address missing");

        goto out;

    }



    crumpled_addr = qdict_crumple(addr, errp);

    if (!crumpled_addr) {

        goto out;

    }










    iv = qobject_input_visitor_new(crumpled_addr);

    visit_type_NFSServer(iv, NULL, &server, &local_error);

    if (local_error) {

        error_propagate(errp, local_error);

        goto out;

    }



out:

    QDECREF(addr);

    qobject_decref(crumpled_addr);

    visit_free(iv);

    return server;

}