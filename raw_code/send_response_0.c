static int send_response(GAState *s, QObject *payload)

{

    const char *buf;

    QString *payload_qstr;

    GIOStatus status;



    g_assert(payload && s->channel);



    payload_qstr = qobject_to_json(payload);

    if (!payload_qstr) {

        return -EINVAL;

    }



    qstring_append_chr(payload_qstr, '\n');

    buf = qstring_get_str(payload_qstr);

    status = ga_channel_write_all(s->channel, buf, strlen(buf));

    QDECREF(payload_qstr);

    if (status != G_IO_STATUS_NORMAL) {

        return -EIO;

    }



    return 0;

}
