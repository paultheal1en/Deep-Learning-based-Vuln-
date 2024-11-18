static int nbd_negotiate_handle_export_name(NBDClient *client, uint32_t length,

                                            uint16_t myflags, bool no_zeroes,

                                            Error **errp)

{

    char name[NBD_MAX_NAME_SIZE + 1];

    char buf[8 + 4 + 124] = "";

    size_t len;

    int ret;



    /* Client sends:

        [20 ..  xx]   export name (length bytes)

     */

    trace_nbd_negotiate_handle_export_name();

    if (length >= sizeof(name)) {

        error_setg(errp, "Bad length received");

        return -EINVAL;

    }

    if (nbd_read(client->ioc, name, length, errp) < 0) {

        error_prepend(errp, "read failed: ");

        return -EINVAL;

    }

    name[length] = '\0';



    trace_nbd_negotiate_handle_export_name_request(name);



    client->exp = nbd_export_find(name);

    if (!client->exp) {

        error_setg(errp, "export not found");

        return -EINVAL;

    }



    trace_nbd_negotiate_new_style_size_flags(client->exp->size,

                                             client->exp->nbdflags | myflags);

    stq_be_p(buf, client->exp->size);

    stw_be_p(buf + 8, client->exp->nbdflags | myflags);

    len = no_zeroes ? 10 : sizeof(buf);

    ret = nbd_write(client->ioc, buf, len, errp);

    if (ret < 0) {

        error_prepend(errp, "write failed: ");

        return ret;

    }



    QTAILQ_INSERT_TAIL(&client->exp->clients, client, next);

    nbd_export_get(client->exp);



    return 0;

}
