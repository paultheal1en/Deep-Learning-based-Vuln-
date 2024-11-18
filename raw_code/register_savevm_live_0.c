int register_savevm_live(const char *idstr,

                         int instance_id,

                         int version_id,

                         SaveLiveStateHandler *save_live_state,

                         SaveStateHandler *save_state,

                         LoadStateHandler *load_state,

                         void *opaque)

{

    SaveStateEntry *se;



    se = qemu_malloc(sizeof(SaveStateEntry));

    pstrcpy(se->idstr, sizeof(se->idstr), idstr);

    se->version_id = version_id;

    se->section_id = global_section_id++;

    se->save_live_state = save_live_state;

    se->save_state = save_state;

    se->load_state = load_state;

    se->opaque = opaque;

    se->vmsd = NULL;



    if (instance_id == -1) {

        se->instance_id = calculate_new_instance_id(idstr);

    } else {

        se->instance_id = instance_id;

    }

    /* add at the end of list */

    TAILQ_INSERT_TAIL(&savevm_handlers, se, entry);

    return 0;

}
