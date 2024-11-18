static void migrate_put_status(QDict *qdict, const char *name,

                               uint64_t trans, uint64_t rem, uint64_t total)

{

    QObject *obj;



    obj = qobject_from_jsonf("{ 'transferred': %" PRId64 ", "

                               "'remaining': %" PRId64 ", "

                               "'total': %" PRId64 " }", trans, rem, total);

    assert(obj != NULL);



    qdict_put_obj(qdict, name, obj);

}
