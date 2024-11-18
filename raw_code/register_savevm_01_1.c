int register_savevm(const char *idstr,

                    int instance_id,

                    int version_id,

                    SaveStateHandler *save_state,

                    LoadStateHandler *load_state,

                    void *opaque)

{

    SaveStateEntry *se, **pse;



    se = qemu_malloc(sizeof(SaveStateEntry));

    if (!se)

        return -1;

    pstrcpy(se->idstr, sizeof(se->idstr), idstr);

    se->instance_id = (instance_id == -1) ? 0 : instance_id;

    se->version_id = version_id;

    se->save_state = save_state;

    se->load_state = load_state;

    se->opaque = opaque;

    se->next = NULL;



    /* add at the end of list */

    pse = &first_se;

    while (*pse != NULL) {

        if (instance_id == -1

                && strcmp(se->idstr, (*pse)->idstr) == 0

                && se->instance_id <= (*pse)->instance_id)

            se->instance_id = (*pse)->instance_id + 1;

        pse = &(*pse)->next;

    }

    *pse = se;

    return 0;

}
