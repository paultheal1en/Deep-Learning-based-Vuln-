 static MachineClass *machine_parse(const char *name)

{

    MachineClass *mc = NULL;

    GSList *el, *machines = object_class_get_list(TYPE_MACHINE, false);



    if (name) {

        mc = find_machine(name);

    }

    if (mc) {


        return mc;

    }

    if (name && !is_help_option(name)) {

        error_report("Unsupported machine type");

        error_printf("Use -machine help to list supported machines!\n");

    } else {

        printf("Supported machines are:\n");

        machines = g_slist_sort(machines, machine_class_cmp);

        for (el = machines; el; el = el->next) {

            MachineClass *mc = el->data;

            if (mc->alias) {

                printf("%-20s %s (alias of %s)\n", mc->alias, mc->desc, mc->name);

            }

            printf("%-20s %s%s\n", mc->name, mc->desc,

                   mc->is_default ? " (default)" : "");

        }

    }




    exit(!name || !is_help_option(name));

}