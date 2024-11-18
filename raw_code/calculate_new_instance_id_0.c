static int calculate_new_instance_id(const char *idstr)

{

    SaveStateEntry *se;

    int instance_id = 0;



    TAILQ_FOREACH(se, &savevm_handlers, entry) {

        if (strcmp(idstr, se->idstr) == 0

            && instance_id <= se->instance_id) {

            instance_id = se->instance_id + 1;

        }

    }

    return instance_id;

}
