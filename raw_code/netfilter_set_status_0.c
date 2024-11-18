static void netfilter_set_status(Object *obj, const char *str, Error **errp)

{

    NetFilterState *nf = NETFILTER(obj);

    NetFilterClass *nfc = NETFILTER_GET_CLASS(obj);



    if (strcmp(str, "on") && strcmp(str, "off")) {

        error_setg(errp, "Invalid value for netfilter status, "

                         "should be 'on' or 'off'");

        return;

    }

    if (nf->on == !strcmp(str, "on")) {

        return;

    }

    nf->on = !nf->on;

    if (nfc->status_changed) {

        nfc->status_changed(nf, errp);

    }

}
