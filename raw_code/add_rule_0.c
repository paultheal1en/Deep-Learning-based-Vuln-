static int add_rule(QemuOpts *opts, void *opaque)

{

    struct add_rule_data *d = opaque;

    BDRVBlkdebugState *s = d->s;

    const char* event_name;

    BlkDebugEvent event;

    struct BlkdebugRule *rule;



    /* Find the right event for the rule */

    event_name = qemu_opt_get(opts, "event");

    if (!event_name || get_event_by_name(event_name, &event) < 0) {

        return -1;

    }



    /* Set attributes common for all actions */

    rule = g_malloc0(sizeof(*rule));

    *rule = (struct BlkdebugRule) {

        .event  = event,

        .action = d->action,

        .state  = qemu_opt_get_number(opts, "state", 0),

    };



    /* Parse action-specific options */

    switch (d->action) {

    case ACTION_INJECT_ERROR:

        rule->options.inject.error = qemu_opt_get_number(opts, "errno", EIO);

        rule->options.inject.once  = qemu_opt_get_bool(opts, "once", 0);

        rule->options.inject.immediately =

            qemu_opt_get_bool(opts, "immediately", 0);

        rule->options.inject.sector = qemu_opt_get_number(opts, "sector", -1);

        break;



    case ACTION_SET_STATE:

        rule->options.set_state.new_state =

            qemu_opt_get_number(opts, "new_state", 0);

        break;



    case ACTION_SUSPEND:

        rule->options.suspend.tag =

            g_strdup(qemu_opt_get(opts, "tag"));

        break;

    };



    /* Add the rule */

    QLIST_INSERT_HEAD(&s->rules[event], rule, next);



    return 0;

}
