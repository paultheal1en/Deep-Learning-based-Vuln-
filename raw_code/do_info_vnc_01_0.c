void do_info_vnc(Monitor *mon, QObject **ret_data)

{

    if (vnc_display == NULL || vnc_display->display == NULL) {

        *ret_data = qobject_from_jsonf("{ 'enabled': false }");

    } else {

        QDict *qdict;

        QList *clist;



        clist = qlist_new();

        if (vnc_display->clients) {

            VncState *client = vnc_display->clients;

            while (client) {

                qdict = do_info_vnc_client(mon, client);

                if (qdict)

                    qlist_append(clist, qdict);

                client = client->next;

            }

        }



        *ret_data = qobject_from_jsonf("{ 'enabled': true, 'clients': %p }",

                                       QOBJECT(clist));

        assert(*ret_data != NULL);



        if (vnc_server_info_put(qobject_to_qdict(*ret_data)) < 0) {

            qobject_decref(*ret_data);

            *ret_data = NULL;

        }

    }

}
