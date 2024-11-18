static SocketAddress *nbd_config(BDRVNBDState *s, QDict *options, Error **errp)

{

    SocketAddress *saddr = NULL;

    QDict *addr = NULL;

    QObject *crumpled_addr = NULL;

    Visitor *iv = NULL;

    Error *local_err = NULL;



    qdict_extract_subqdict(options, &addr, "server.");

    if (!qdict_size(addr)) {

        error_setg(errp, "NBD server address missing");

        goto done;

    }



    crumpled_addr = qdict_crumple(addr, errp);

    if (!crumpled_addr) {

        goto done;

    }











    iv = qobject_input_visitor_new(crumpled_addr);

    visit_type_SocketAddress(iv, NULL, &saddr, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto done;

    }



done:

    QDECREF(addr);

    qobject_decref(crumpled_addr);

    visit_free(iv);

    return saddr;

}