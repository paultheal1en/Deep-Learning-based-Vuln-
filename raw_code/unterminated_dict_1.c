static void unterminated_dict(void)

{

    QObject *obj = qobject_from_json("{'abc':32", NULL);

    g_assert(obj == NULL);

}
