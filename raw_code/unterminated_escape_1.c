static void unterminated_escape(void)

{

    QObject *obj = qobject_from_json("\"abc\\\"", NULL);

    g_assert(obj == NULL);

}
