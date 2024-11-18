static SaveStateEntry *find_se(const char *idstr, int instance_id)

{

    SaveStateEntry *se;



    QTAILQ_FOREACH(se, &savevm_handlers, entry) {

        if (!strcmp(se->idstr, idstr) &&

            instance_id == se->instance_id)

            return se;

    }

    return NULL;

}
