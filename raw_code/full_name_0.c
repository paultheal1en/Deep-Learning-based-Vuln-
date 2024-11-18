static const char *full_name(QObjectInputVisitor *qiv, const char *name)

{

    StackObject *so;

    char buf[32];



    if (qiv->errname) {

        g_string_truncate(qiv->errname, 0);

    } else {

        qiv->errname = g_string_new("");

    }



    QSLIST_FOREACH(so , &qiv->stack, node) {

        if (qobject_type(so->obj) == QTYPE_QDICT) {

            g_string_prepend(qiv->errname, name);

            g_string_prepend_c(qiv->errname, '.');

        } else {

            snprintf(buf, sizeof(buf), "[%u]", so->index);

            g_string_prepend(qiv->errname, buf);

        }

        name = so->name;

    }



    if (name) {

        g_string_prepend(qiv->errname, name);

    } else if (qiv->errname->str[0] == '.') {

        g_string_erase(qiv->errname, 0, 1);

    } else {

        return "<anonymous>";

    }



    return qiv->errname->str;

}
