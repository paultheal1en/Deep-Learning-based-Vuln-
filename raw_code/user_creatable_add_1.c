Object *user_creatable_add(const QDict *qdict,

                           Visitor *v, Error **errp)

{

    char *type = NULL;

    char *id = NULL;

    Object *obj = NULL;

    Error *local_err = NULL, *end_err = NULL;

    QDict *pdict;



    pdict = qdict_clone_shallow(qdict);



    visit_start_struct(v, NULL, NULL, 0, &local_err);

    if (local_err) {

        goto out;

    }



    qdict_del(pdict, "qom-type");

    visit_type_str(v, "qom-type", &type, &local_err);

    if (local_err) {

        goto out_visit;

    }



    qdict_del(pdict, "id");

    visit_type_str(v, "id", &id, &local_err);

    if (local_err) {

        goto out_visit;

    }



    obj = user_creatable_add_type(type, id, pdict, v, &local_err);

    if (local_err) {

        goto out_visit;

    }



 out_visit:

    visit_end_struct(v, &end_err);

    if (end_err) {

        error_propagate(&local_err, end_err);

        if (obj) {

            user_creatable_del(id, NULL);

        }

        goto out;

    }



out:

    QDECREF(pdict);

    g_free(id);

    g_free(type);

    if (local_err) {

        error_propagate(errp, local_err);

        object_unref(obj);

        return NULL;

    }

    return obj;

}
